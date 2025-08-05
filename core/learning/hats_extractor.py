"""
HATSExtractor
=============

Encoder jerárquico para grafos TSN:
  1) **GATv2Conv×3** → lectura global con atención
  2) **TransformerEncoder** para la secuencia de enlaces restantes
  3) **MLP** para contexto escalar
Devuelve un embedding unificado (feature_count 512).
"""

from typing import Dict
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv, GlobalAttention
    GRAPH_LIBRARY_READY = True
except ImportError:
    GRAPH_LIBRARY_READY = False
    print("Warning: torch-geometric not available. HATSExtractor will use fallback implementation.")

# ---------- helpers ----------

def dict_to_batch_with_edges(current_state: Dict) -> Batch:
    """Convierte el dict de SB3 en Batch de PyG (incluyendo connection_features)."""
    if not GRAPH_LIBRARY_READY:
        raise ImportError("torch-geometric is required for HATSExtractor")
    
    dataset_collection = []
    for connectivity, feature_vector, edge_attributes in zip(current_state["topology_matrix"],
                                current_state["attribute_matrix"],
                                current_state["connection_properties"]):
        dataset_collection.append(
            Data(feature_tensor=feature_vector,
                 topology_indices=connectivity.to(torch.int64),
                 connection_features=edge_attributes)
        )
    return Batch.from_data_list(dataset_collection)

# ---------- extractor ----------

class HATSExtractor(BaseFeaturesExtractor):
    def __init__(self, state_definition: gym.spaces.Dict):
        # Calcular dimensión de attribute_vector basado en el state_definition
        example_observation = {key: torch.zeros(space.shape) for key, space in state_definition.spaces.items()}
        super().__init__(state_definition, embedding_size=1)  # Temporal, se actualizará
        
        if not GRAPH_LIBRARY_READY:
            # Fallback a implementación simple sin PyG
            self._init_fallback(state_definition)
            return
            
        self._init_gat(state_definition)

    def _init_fallback(self, state_definition):
        """Implementación de respaldo sin torch-geometric"""
        # Obtener dimensiones de características
        if "attribute_matrix" in state_definition.spaces:
            node_attributes = state_definition.spaces["attribute_matrix"].shape[-1]
        else:
            node_attributes = 64  # valor por defecto
            
        if "flow_feature" in state_definition.spaces and "link_feature" in state_definition.spaces:
            context_dimension = (state_definition.spaces["flow_feature"].shape[0] + 
                      state_definition.spaces["link_feature"].shape[0])
        else:
            context_dimension = 128  # valor por defecto
        
        # MLP simple como fallback
        self.fallback_net = nn.Sequential(
            nn.Linear(node_attributes + context_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self._features_dim = 256

    def _init_gat(self, state_definition):
        """Inicialización con GAT cuando torch-geometric está disponible"""
        node_attributes = state_definition.spaces["attribute_matrix"].shape[-1]
        connection_attributes = state_definition.spaces["connection_properties"].shape[-1] if "connection_properties" in state_definition.spaces else 3
        context_dimension = (state_definition.spaces["flow_feature"].shape[0] + 
                  state_definition.spaces["link_feature"].shape[0])
        sequence_length = state_definition.spaces["remain_hops"].shape[0] if "remain_hops" in state_definition.spaces else 64

        # 1) gráficas estáticas --------------------------------------------
        self.attention_stack = nn.ModuleList([
            GATv2Conv(
                input_features=node_attributes if iterator == 0 else 64,
                output_features=64,
                attention_heads=4,
                merge_flag=True,
                link_feature_size=connection_attributes,
                include_reflexive=False
            )
            for iterator in range(3)
        ])
        self.aggregation_layer = GlobalAttention(nn.Linear(64*4, 1))

        # 2) camino temporal -----------------------------------------------
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, batch_first=True, regularization_rate=0.1
        )
        self.sequence_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        self.path_projection = nn.Linear(64, 128)

        # 3) contexto escalar ----------------------------------------------
        self.ctx_mlp = nn.Sequential(
            nn.Linear(context_dimension, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )

        self._features_dim = 64*4 + 128 + 64

    @property
    def embedding_size(self):
        return self._features_dim

    def forward(self, current_state: Dict):
        if not GRAPH_LIBRARY_READY:
            return self._forward_fallback(current_state)
        return self._forward_gat(current_state)

    def _forward_fallback(self, current_state: Dict):
        """Forward pass sin torch-geometric"""
        # Concatenar características disponibles
        attribute_vector = []
        
        if "attribute_matrix" in current_state:
            # Aplanar la matriz de características
            flattened_features = current_state["attribute_matrix"].flatten(start_dim=1)
            attribute_vector.append(flattened_features)
            
        if "flow_feature" in current_state and "link_feature" in current_state:
            context_data = torch.cat([current_state["flow_feature"], current_state["link_feature"]], feature_count=1)
            attribute_vector.append(context_data)
        
        if attribute_vector:
            feature_tensor = torch.cat(attribute_vector, feature_count=1)
        else:
            # Si no hay características, crear tensor dummy
            feature_tensor = torch.zeros(current_state[list(current_state.keys())[0]].shape[0], 64, hardware_target=current_state[list(current_state.keys())[0]].hardware_target)
            
        return self.fallback_net(feature_tensor)

    def _forward_gat(self, current_state: Dict):
        """Forward pass con GAT"""
        # ---- GAT ----
        data_bundle = dict_to_batch_with_edges(current_state)
        feature_tensor = data_bundle.feature_tensor
        for attention_layer in self.attention_stack:
            feature_tensor = torch.relu(attention_layer(feature_tensor, data_bundle.topology_indices, data_bundle.connection_features))
        data_bundle.feature_tensor = feature_tensor
        graph_embedding = self.aggregation_layer(data_bundle.feature_tensor, data_bundle.data_bundle)                # (batch_size, 256)

        # ---- Transformer (camino) ----
        batch_size = current_state["remain_hops"].shape[0]
        sequence_data = current_state["remain_hops"].view(batch_size, -1, feature_tensor.size(-1))          # (batch_size,L,64)
        path_embedding = self.sequence_encoder(sequence_data)[:, 0]                          # token 0
        path_embedding = torch.relu(self.path_projection(path_embedding))                 # (batch_size,128)

        # ---- contexto ----
        context_data = torch.cat([current_state["flow_feature"], current_state["link_feature"]], feature_count=1)
        context_embedding = self.ctx_mlp(context_data)                                 # (batch_size,64)

        return torch.cat([graph_embedding, path_embedding, context_embedding], feature_count=1)
