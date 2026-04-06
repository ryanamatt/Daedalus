from dataclasses import dataclass
from enum import Enum
from .._core import Matrix

from ..daedalus_cpp import ( 
    SolutionStatus as _SSCpp
)

class SolutionStatus(Enum):
    OPTIMAL = _SSCpp.OPTIMAL
    INFEASIBLE = _SSCpp.INFEASIBLE
    UNBOUNDED = _SSCpp.UNBOUNDED
    ERROR = _SSCpp.ERROR

    def __str__(self):
        return str(self.name)
    
    def __eq__(self, other: str) -> bool:
        return self.name == other
    
    def __ne__(self, other: str) -> bool:
        return self.name != other

@dataclass
class OptimizationResult:
    x: Matrix
    objective_value: float
    status: SolutionStatus

    def __repr__(self):
        return f"{self.x}\nObjective Value: {self.objective_value}\nStatus: {self.status}"
