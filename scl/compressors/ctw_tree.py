from scl.compressors.ctw_node import CTWNode
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray

from collections import deque
import numpy as np

# TODO: Original paper describes storage complexity as linear in D...but the size of the tree is ~ 2^D
# TODO: (related to above) do path pruning to save memory?

class CTWTree():
    """
    Represents the CTW tree

    Store the root node, context, and snapshot
    """
    root: CTWNode = None                        # root node of CTW tree
    tree_height: int = None
    current_context: BitArray = None            # context (past symbols seen) for current state of CTW tree
    snapshot: list = None                       # list of nodes that were updated so that we can revert the update
    get_snapshot: bool = None                   # flag telling us whether to save what nodes we've updated

    def __init__(self, tree_height: int, past_context: BitArray):
        """
        Initialize the CTW tree with given height and context
        """

        assert len(past_context) == tree_height

        self.tree_height = tree_height
        self.current_context = deque(past_context, maxlen=tree_height)

        # Populate the nodes of the tree
        root = CTWNode(id=BitArray())
        queue = deque([(root, tree_height)])
        while queue:
            node, depth = queue.popleft()
            if depth > 0:
                node.left_child = CTWNode(id=node.id + BitArray("0"))
                node.right_child = CTWNode(id=node.id + BitArray("1"))
                queue.append((node.left_child, depth - 1))
                queue.append((node.right_child, depth - 1))
        self.root = root

    def print_tree(self):
        """
        Call print_node on the root node to print the CTW tree
        """

        self.root.print_node()
    
    def update_tree(self, sequence: BitArray):
        """
        Update the CTW tree with the given sequence of symbols
        and updates the context accordingly
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
            node.a, node.b, node.kt_prob_log2, node.node_prob_log2 = prev_state

        # Clear the snapshot after completing revert
        self.snapshot = []

    def get_root_prob(self) -> float:
        """
        Get the node probability of the root node

        prob = 2**prob_log2
        """

        return 2**self.root.node_prob_log2

    # TODO: This sometimes returns probability 0
    # Or infinite probability -- probably should add checks/asserts?
    def get_symbol_prob(self, symbol: bool) -> float:
        """
        Compute the probability of seeing the given symbol based on the current state of the CTW tree

        P(symbol | context) = P(symbol, context) / P(context)
        """

        # Save the updated nodes so we can revert them
        self.snapshot = []
        self.get_snapshot = True

        # Get the probability of the context
        context_prob_log2 = self.root.node_prob_log2

        # Update the CTW tree with the given symbol
        self.update_tree_symbol(symbol)

        # Get the probability of the combined symbol and context
        symbol_context_prob_log2 = self.root.node_prob_log2

        # Compute the probability of the symbol given the context
        symbol_prob_log2 = symbol_context_prob_log2 - context_prob_log2

        # Undo the changes made (revert to before we added the given symbol)
        self.revert_tree()
        self.get_snapshot = False

        return 2**symbol_prob_log2

    def update_tree_symbol(self, next_symbol: bool):
        """
        NOTE: Does NOT update self.current_context
        Update the CTW tree with the given symbol by traversing the branch corresponding to the current context
        starting from the leaf node of the branch and updating the nodes towards the root
        """
        assert next_symbol == 0 or next_symbol == 1
        self._update_node(node=self.root, context=self.current_context, symbol=next_symbol)
    
    def _update_node(self, node: CTWNode, context: deque, symbol: bool):
        # If we have reached the end of the context, this is as far as we traverse
        # Update the snapshot of changed nodes (if needed), and update the node
        if len(context) == 0:
            if self.get_snapshot:
                self.snapshot.append((node, (node.a, node.b, node.kt_prob_log2, node.node_prob_log2)))
            node.kt_update_log2(symbol)
            return

        # Since the context is a deque, it is more effecient to pop then re-add than it is to access by index
        # Store and remove the latest symbol of the context
        latest_context_symbol = context.pop()

        # Update the child (based on the latest symbol of the context) of the node first
        self._update_node(node=node.get_child(latest_context_symbol), context=context, symbol=symbol)

        # Re-add the symbol removed from the context
        context.append(latest_context_symbol)

        # Then update the snapshot of changed nodes (if needed), and update the node
        if self.get_snapshot:
            self.snapshot.append((node, (node.a, node.b, node.kt_prob_log2, node.node_prob_log2)))
        node.kt_update_log2(symbol)
        
    def update_context(self, context: BitArray):
        assert len(context) <= len(self.current_context)
        # Update the context
        # Remove the beginning of the context
        self.current_context.extend(context)

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
    test_tree.update_context(BitArray("0"))
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