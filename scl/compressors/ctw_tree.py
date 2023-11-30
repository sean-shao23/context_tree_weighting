from scl.compressors.ctw_node import CTWNode
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
import copy
import numpy as np

# TODO: Original paper describes storage complexity as linear in D...but the size of the tree is ~ 2^D
# TODO: (related to above) do path pruning to save memory?

class CTWTree():
    """
    Represents the CTW tree

    Store the root node, context, and snapshot
    """
    root: CTWNode = None                        # root node of CTW tree
    current_context: BitArray = None            # context (past symbols seen) for current state of CTW tree
    snapshot: list = None                       # list of nodes that were updated so that we can revert the update
    get_snapshot: bool = None                   # flag telling us whether to save what nodes we've updated

    def __init__(self, tree_height: int, past_context: BitArray):
        """
        Initialize the CTW tree with given height and context
        """

        assert len(past_context) == tree_height

        self.current_context = past_context

        # Call recursive function self.gen_tree() to populate the nodes of the tree
        self.root = self.gen_tree(depth=tree_height, node_context=BitArray())

    def print_tree(self):
        """
        Call print_node on the root node to print the CTW tree
        """

        self.root.print_node()

    def gen_tree(self, depth: int, node_context: BitArray) -> CTWNode:
        """
        Generate the subtree of given depth
        """

        # If depth is 0, node has no children (is a leaf of the CTW tree)
        if depth == 0:
            return CTWNode(id=node_context, left_child=None, right_child=None)
        
        # Generate the left and right subtrees
        left_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray("0"))
        right_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray("1"))

        # Create the root node for this subtree
        return CTWNode(id=node_context, left_child=left_child, right_child=right_child)
    
    def update_tree(self, sequence: BitArray):
        """
        Update the CTW tree with the given sequence of symbols
        """

        for symbol in sequence:
            self.update_tree_symbol(symbol)
            self.update_context([symbol])

    def revert_tree(self):
        """
        Revert the tree according to snapshot
        """

        for node, prev_state in self.snapshot:
            assert type(node) == CTWNode
            assert type(prev_state) == CTWNode

            node.a = prev_state.a
            node.b = prev_state.b
            node.kt_prob_as_ints = prev_state.kt_prob_as_ints
            node.node_prob_as_ints = prev_state.node_prob_as_ints

        # Clear the snapshot after completing revert
        self.snapshot = []

    def get_root_prob(self) -> float:
        """
        Get the node probability of the root node as a floating point value

        prob = numerator / (2^[log2(denominator)])
        """

        num, denom_log = self.root.node_prob_as_ints
        return num / (2**denom_log)

    # TODO: Will this return probability 0? If so...is that due to precision issues?
    def get_symbol_prob(self, symbol: bool) -> float:
        """
        Compute the probability of seeing the given symbol based on the current state of the CTW tree

        P(symbol | context) = P(symbol, context) / P(context)
        """

        # Save the updated nodes so we can revert them
        self.snapshot = []
        self.get_snapshot = True

        # Update the CTW tree with the given symbol
        # We don't call update_tree_symbol() as that would update the context
        self._update_node(node=self.root, context=self.current_context, symbol=symbol)

        # Compute the root probability after adding the given symbol
        new_tree_prob_num, new_tree_prob_denom_log = self.root.node_prob_as_ints

        # Undo the changes made (revert to before we added the given symbol)
        self.revert_tree()
        self.get_snapshot = False

        # Compute the actual root probability (after reverting)
        tree_prob_num, tree_prob_denom_log = self.root.node_prob_as_ints

        # Compute new_prob/actual_prob
        denom_log = new_tree_prob_denom_log - tree_prob_denom_log
        symbol_prob = (new_tree_prob_num / tree_prob_num) / (2**denom_log)
        
        # Return the ratio of the probabilities 
        return symbol_prob

    def update_tree_symbol(self, next_symbol: bool):
        """
        Update the CTW tree with the given symbol

        P(symbol | context) = P(symbol, context) / P(context)
        """

        # Call recursive function _update_node() to update the nodes of the tree
        self._update_node(node=self.root, context=self.current_context, symbol=next_symbol)

        
    def update_context(self, context: list):
        assert len(context) <= len(self.current_context)
        # Update the context
        # Remove the beginning of the context
        self.current_context = self.current_context[len(context):] + BitArray(context)

    def _update_node(self, node: CTWNode, context: str, symbol: bool):
        """
        First update the children of the given node
        then update the node itself

        We traverse the tree according to the context, so only the path of the tree
        corresponding to the context needs to be update
        """
        # If the context length is 0, this a leaf node
        if len(context) == 0:
            # Add node to snapshot (if needed)
            if self.get_snapshot:
                self.snapshot.append((node, copy.deepcopy(node)))

            # Update the node's counts and probabilities
            node.kt_update(symbol)
            return

        # Update the corresponding child (based on what's left of the context to traverse)
        self._update_node(node=node.get_child(context[-1]), context=context[:-1], symbol=symbol)

        # Add node to snapshot (if needed)
        if self.get_snapshot:
            self.snapshot.append((node, copy.deepcopy(node)))

        # Update the node's counts and probabilities
        node.kt_update(symbol)

# TODO: These tests only test with tree depth 3
# We should probably add tests for other depths

def test_ctw_tree_generation():
    # Depth 3 CTW tree with no symbols (but context of 1, 1, 0) added
    # Should have default root probability 1
    test_tree = CTWTree(tree_height=3, past_context=BitArray("110"))
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        1,
    )

    # CTW tree after adding symbols 0, 1, 0, 0, 1, 1, 0
    test_tree.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        7/2048,
    )
    
    # CTW tree after adding symbol 0
    test_tree.update_tree_symbol(0)
    test_tree.update_context([0])
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        153/65536,
    )

def test_ctw_tree_probability():
    # Depth 3 CTW tree after adding symbols 0, 1, 0, 0, 1, 1, 0
    # With context of 1, 1, 0
    test_tree = CTWTree(tree_height=3, past_context=BitArray("110"))
    test_tree.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test_tree.get_symbol_prob(0),
        (153/65536)/(7/2048)
    )

    # CTW tree probability state should be unchanged after
    # computing the probability of seeing a symbol
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        7/2048,
    )

    # CTW tree probability for 1 should be (1 - probability of 0)
    np.testing.assert_almost_equal(
        test_tree.get_symbol_prob(1),
        (71/65536)/(7/2048)
    )