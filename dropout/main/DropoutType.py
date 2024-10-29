from enum import Enum

import dropout.data.dictionary.DropoutDataField as DDF
import dropout.strategies.rotation.RotationDropoutHandler as RDH
import dropout.strategies.canonical.CanonicalDropoutHandler as CDH
import dropout.strategies.entangling.EntanglingDropoutHandler as EDH 
import dropout.strategies.canonical_forward.CanonicalForwardDropoutHandler as CFDH

class DropoutType(Enum):
    """
    The `DropoutType` class defines various types of dropout strategies that can be applied to 
    quantum circuits. These dropout strategies are designed to probabilistically remove gates 
    from the circuit according to specific rules, which can enhance generalization and reduce 
    the complexity of the circuit. Each dropout type corresponds to a unique strategy, and this 
    class facilitates mapping between the dropout type and its respective implementations through 
    `DropoutHandler` objects.

    There are two main forms of dropouts implemented in this class:

    1) **Novel Dropouts**: 
       - To introduce a new dropout strategy:
         * Define a new field in the Enum for the dropout type.
         * Create a class extending `DropoutHandler` that implements the dropout behavior.
         * Modify the `dropout_to_handlers` method to include the new Enum field, returning a list 
           with an instance of the new class.

    2) **Cascade Dropouts**: 
       - To create a composite dropout consisting of existing dropouts:
         * Define a new field in the Enum for the composite dropout.
         * Adjust the `dropout_to_handlers` method to return a list of `DropoutHandler` objects 
           that represent the sequential dropout strategies to be applied.

    For example, if dropout `C` is a combination of dropouts `A` and `B`, a new Enum field 'C' 
    would be defined, returning a list like `[ADropoutHandler, BDropoutHandler]` in the 
    `dropout_to_handlers` method.

    Enum Fields:
    ------------
    CANONICAL : str
        A dropout strategy applying dropout to basic gates with specific collateral rules.
    CANONICAL_FORWARD : str
        A variation of the canonical dropout, which only removes gates in the forward direction.
    ROTATION : str
        A dropout strategy specifically for rotation (basic) gates.
    ENTANGLING : str
        A dropout strategy that targets entangling gates.
    INDEPENDENT : str
        A composite dropout that applies rotation dropout followed by entangling dropout.
    NO_DROPOUT : str
        A strategy indicating that no dropout should be applied.

    Methods:
    --------
    dropout_to_handlers(dropout_data)
        Maps the specified dropout type to a corresponding list of `DropoutHandler` objects.
        Can return a single handler or multiple handlers in the case of composite dropouts.
    """

    CANONICAL = "CANONICAL"
    CANONICAL_FORWARD = "CANONICAL_FORWARD"
    ROTATION = "ROTATION"
    ENTANGLING = "ENTANGLING"
    INDEPENDENT = "INDEPENDENT"
    NO_DROPOUT = "NO_DROPOUT"

    def type_to_handlers(dropout_data):
        """
        Maps the specified dropout type to its corresponding `DropoutHandler` objects.

        The method uses a match statement to identify the appropriate dropout strategy based on the 
        provided `dropout_data`. Depending on the dropout type, different handlers are instantiated 
        to manage the dropout logic. Some types, such as `INDEPENDENT`, represent composite 
        strategies that return a list of multiple handlers to be executed sequentially.

        Parameters:
        -----------
        dropout_data : DropoutData
            A DropoutData object passed down from the caller.

        Returns:
        --------
        list
            A list of instantiated `DropoutHandler` objects corresponding to the provided dropout type.
            The list may contain a single handler or multiple handlers for composite dropouts.
        
        Notes:
        ------
        - The match statement checks the dropout type and returns the relevant handler(s). For example,
          the `INDEPENDENT` type returns both rotation and entangling dropout handlers to be applied 
          in sequence.
        - In case of `NO_DROPOUT`, an empty list is returned, indicating that no gates will be dropped.
        """
        
        match dropout_data.get_field(DDF.DropoutDataField.TYPE): 
            case DropoutType.ROTATION:  
                # Applies dropout to rotation gates (basic gates) with specified probability.
                return [RDH.RotationDropoutHandler(dropout_data)]
            
            case DropoutType.ENTANGLING:  
                # Applies dropout to entangling gates with specified probability.
                return [EDH.EntanglingDropoutHandler(dropout_data)]

            case DropoutType.CANONICAL:  
                # Applies dropout to basic gates with specific probability and collateral rules.
                return [CDH.CanonicalDropoutHandler(dropout_data)]
            
            case DropoutType.CANONICAL_FORWARD:  
                # Applies dropout to basic gates with specific probability, but only in forward direction.
                return [CFDH.CanonicalForwardDropoutHandler(dropout_data)]
            
            case DropoutType.INDEPENDENT:  
                # Applies a ROTATION dropout followed by an ENTANGLING dropout sequentially.
                dropout_data_entangling = dropout_data
                dropout_data.set_field(DDF.DropoutDataField.TYPE, DropoutType.ROTATION)

                if 'p_ent' in dropout_data.get_field(DDF.DropoutDataField.ADDITIONAL_DATA).keys():
                    p_drop_entangling = dropout_data.get_field(DDF.DropoutDataField.ADDITIONAL_DATA)['p_ent']
                    dropout_data_entangling.set_field(DDF.DropoutDataField.DROPOUT_PROB, p_drop_entangling) 
                    
                dropout_data_entangling.set_field(DDF.DropoutDataField.TYPE, DropoutType.ENTANGLING)

                return [RDH.RotationDropoutHandler(dropout_data), 
                        EDH.EntanglingDropoutHandler(dropout_data_entangling)]

            case DropoutType.NO_DROPOUT:
                return []
            
            case _:
                return []


