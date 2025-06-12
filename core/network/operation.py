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
    latest_time: int  # must equal to gating time if enable gating
    end_time: int
    # Nuevos campos para análisis y visualización
    reception_time: Optional[int] = None
    
    # Solo se conserva la espera por *min-gap* (controlada por RL)
    min_gap_wait: int = 0
    
    # Campos para decisiones de RL (eliminado gcl_cycle_opt)
    guard_factor: float = 1.0
    min_gap_value: float = 1.0
    conflict_strategy: int = 0

    def __post_init__(self):
        if self.gating_time is not None:
            assert self.start_time <= self.gating_time == self.latest_time < self.end_time, \
                   "Invalid Operation: desajuste temporal"
        else:
            assert self.start_time <= self.latest_time < self.end_time, "Invalid Operation"
        
        # Calcular automáticamente el tiempo de recepción completa si no se proporciona
        if self.reception_time is None:
            # Tomar en cuenta retardo de propagación y procesamiento de recepción
            self.reception_time = self.end_time + Net.DELAY_PROP + Net.DELAY_PROC_RX

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
        self.end_time += other
        # earliest_time ya no existe o se alinea con start_time
        # Actualizar reception_time también
        if self.reception_time is not None:
             self.reception_time += other
        return self

    def __repr__(self):
        if self.gating_time is not None:
            return (f"Operation(start={self.start_time}, gate={self.gating_time}, "
                    f"end={self.end_time}, rcv={self.reception_time})")
        return (f"Operation(start={self.start_time}, latest={self.latest_time}, "
                f"end={self.end_time}, rcv={self.reception_time})")

    # ➕ Utilidad
    @property
    def duration(self):
        """Duración efectiva de la transmisión (end − start)"""
        return self.end_time - self.start_time

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
             otherwise, it returns the offset that `operation1` should add.
             Notice that the adding the returned offset might make `operation` out of period.
    """
    operation1, period1 = operation1
    operation2, period2 = operation2

    assert (operation1.start_time >= 0) and (operation1.end_time <= period1)
    assert (operation2.start_time >= 0) and (operation2.end_time <= period2)

    hyper_period = math.lcm(period1, period2)
    alpha = hyper_period // period1
    beta = hyper_period // period2

    operation_lhs = copy.deepcopy(operation1)

    for _ in range(alpha):

        operation_rhs = copy.deepcopy(operation2)
        for _ in range(beta):
            if (operation_lhs.start_time <= operation_rhs.start_time < operation_lhs.end_time) or \
                    (operation_rhs.start_time <= operation_lhs.start_time < operation_rhs.end_time):
                # Desplazamiento bruto necesario para eliminar la colisión
                raw_offset = operation_rhs.end_time - operation_lhs.start_time

                # Reducirlo al mínimo compatible con ambos períodos
                gcd_period = math.gcd(period1, period2)
                offset = raw_offset % gcd_period

                # Si el módulo es 0 (raw_offset múltiplo del mcd) desplazamos un ciclo completo
                return offset if offset != 0 else gcd_period
            operation_rhs.add(period2)

        operation_lhs.add(period1)
    return None
