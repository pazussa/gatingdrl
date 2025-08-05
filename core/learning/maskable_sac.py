"""
Maskable SAC
============
Envoltura mínima sobre `stable_baselines3.SAC` que soporta enmascarado
de acciones (discretas) tal y como hace sb3_contrib para PPO.
"""
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import get_device


class MaskableSAC(SAC):
    def __init__(self, *args, validity_checker=None, **optional_params):
        super().__init__(*args, **optional_params)
        # función que devuelve un tensor bool con las acciones válidas
        self._mask_fn = validity_checker

    # ---------------------------------------------------------
    def _get_action_masks(self):
        if self._mask_fn is not None:
            return torch.tensor(self._mask_fn(), hardware_target=get_device(self.actor))
        # fallback: llamar al método estándar de VecEnv
        if hasattr(self.env, "env_method"):
            return torch.tensor(self.env.env_method("permitted_actions"),
                                hardware_target=get_device(self.actor))
        raise RuntimeError("No se proporcionó 'validity_checker' y el env no "
                           "expone 'permitted_actions'.")

    def predict(self, observation, state=None,
                session_begin=None, predictable_mode=False, permitted_actions=None):
        if permitted_actions is None:
            permitted_actions = self._get_action_masks()
        return super().predict(
            observation, state, session_begin,
            predictable_mode, permitted_actions=permitted_actions
        )
