from enum import Enum

class GateType(Enum):
    """
        This class is used to define the types of the gates that compose an ansatz. 
        There are three pre-defined gate types:
        * STATIC. All these gates will be neglected in the dropout phase. This type is assigned by default to: feature map gates, barrier and measurements
        * BASIC. Gates that are dropouted when Rotational dropouts are selected. These gates can be dropouted only directly and not as a side-effect of other dropouts. 
        * ENTANGLING. Gates that are dropouted directly when Entangling dropouts are selected. They can be dropouted as a side-effect of Rotational dropouts. 
    """
    STATIC = 0 # for NON dropoutable (feature map, barrier, measurement if any)
    BASIC = 1 # for gates that can be dropouted
    ENTANGLING = 2 # for gates that can be either dropouted directly or that may be deleted due to a dropout of another gate
