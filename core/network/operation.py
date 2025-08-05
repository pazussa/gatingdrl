import copy
from dataclasses import dataclass
import math
from typing import Optional
import numpy as np

from core.network.net import Net  # Añadido para acceder a Net.DELAY_PROC_RX



@dataclass
class Operation:
    start_time: int
    gating_time: Optional[int]
    latest_time: int  # must equal to time_synchronization time if enable time_synchronization
    completion_instant: int
    # Nuevos campos para análisis y visualización
    reception_time: Optional[int] = None
    
    # Solo se conserva la espera por *min-gap* (controlada por RL)
    min_gap_wait: int = 0
    
    # Campos para decisiones de RL (eliminado gcl_cycle_opt)
    protection_multiplier: float = 1.0
    min_gap_value: float = 1.0
    conflict_strategy: int = 0

    def __post_init__(self):
        if self.gating_time is not None:
            assert self.start_time <= self.gating_time == self.latest_time < self.completion_instant, \
                   "Invalid Operation: desajuste temporal"
        else:
            assert self.start_time <= self.latest_time < self.completion_instant, "Invalid Operation"
        
        # Calcular automáticamente el tiempo de recepción completa si no se proporciona
        if self.reception_time is None:
            # Tomar en cuenta retardo de propagación y procesamiento de recepción
            self.reception_time = self.completion_instant + Net.DELAY_PROP + Net.DELAY_PROC_RX

    def add(self, other: int | np.integer):
        """
        Desplaza la operación en ``other`` µs.
        Acepta enteros python normales **o** cualquier subtipo de ``np.integer``.
        """
        # Convertir numpy.int* a int nativo para evitar fallos de tipo
        if isinstance(other, np.integer):
            other = int(other)
        assert isinstance(other, int), "Operation.add() espera un entero"

        self.start_time += other
        if self.gating_time is not None:
            self.gating_time += other
        self.latest_time += other
        self.completion_instant += other
        # earliest_time ya no existe o se alinea con start_time
        # Actualizar reception_time también
        if self.reception_time is not None:
             self.reception_time += other
        return self

    def __repr__(self):
        if self.gating_time is not None:
            return (f"Operation(start={self.start_time}, gate={self.gating_time}, "
                    f"end={self.completion_instant}, rcv={self.reception_time})")
        return (f"Operation(start={self.start_time}, maximum_time={self.latest_time}, "
                f"end={self.completion_instant}, rcv={self.reception_time})")

    # ➕ Utilidad
    @property
    def duration(self):
        """Duración efectiva de la transmisión (end − start)"""
        return self.completion_instant - self.start_time

    # Nueva propiedad para obtener el desglose de esperas
    @property
    def wait_breakdown(self):
        """Retorna un diccionario con el desglose de las causas de espera"""
        if self.gating_time is None or self.gating_time <= self.start_time:
            return {}
        
        total_wait = self.gating_time - self.start_time
        result = {
            'total': total_wait,
            'min_gap': self.min_gap_wait,
            'other': total_wait - self.min_gap_wait
        }
        return result

    # Eliminar la propiedad jitter_percent
    
    # Método para registrar el período asociado
    def set_period(self, period: int):
        """Almacena el período asociado a esta operación para cálculos posteriores"""
        self._period = period
        return self

#
def check_operation_isolation(operation1: tuple[Operation, int],
                              operation2: tuple[Operation, int]) -> Optional[int]:
    """

    :param operation1:
    :param operation2:
    :return: None if isolation constraint is satisfied,
             otherwise, it returns the time_adjustment that `operation1` should add.
             Notice that the adding the returned time_adjustment might make `operation` out of period.
    """
    import time
    start_time = time.time()
    
    operation1, period1 = operation1
    operation2, period2 = operation2

    assert (operation1.start_time >= 0) and (operation1.completion_instant <= period1)
    assert (operation2.start_time >= 0) and (operation2.completion_instant <= period2)

    hyper_period = math.lcm(period1, period2)
    alpha = hyper_period // period1
    beta = hyper_period // period2
    
    # Registrar métricas de hiperperíodo
    try:
        from tools.complexity_metrics import get_metrics
        metrics = get_metrics()
        metrics.record_hyperperiod_calculation(period1, period2, hyper_period)
    except ImportError:
        pass  # Métricas opcionales

    operation_lhs = copy.deepcopy(operation1)

    for _ in range(alpha):

        operation_rhs = copy.deepcopy(operation2)
        for _ in range(beta):
            if (operation_lhs.start_time <= operation_rhs.start_time < operation_lhs.completion_instant) or \
                    (operation_rhs.start_time <= operation_lhs.start_time < operation_rhs.completion_instant):
                # Desplazamiento bruto necesario para eliminar la colisión
                raw_offset = operation_rhs.completion_instant - operation_lhs.start_time

                # Reducirlo al mínimo compatible con ambos períodos
                gcd_period = math.gcd(period1, period2)
                time_adjustment = raw_offset % gcd_period

                # Si el módulo es 0 (raw_offset múltiplo del mcd) desplazamos un ciclo completo
                result = time_adjustment if time_adjustment != 0 else gcd_period
                
                # Registrar tiempo de ejecución
                try:
                    end_time = time.time()
                    metrics.record_isolation_check_time(end_time - start_time)
                except:
                    pass
                
                return result
            operation_rhs.add(period2)

        operation_lhs.add(period1)
    
    # Registrar tiempo de ejecución para caso sin conflicto
    try:
        end_time = time.time()
        metrics.record_isolation_check_time(end_time - start_time)
    except:
        pass
        
    return None
