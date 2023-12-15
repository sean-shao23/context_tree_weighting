from __future__ import annotations
from math import log2
import numpy as np
from scl.utils.tree_utils import BinaryNode

class CTWNode(BinaryNode):
    """
    Represents a node of the CTW tree

    NOTE: BinaryNode class already has left_child, right_child, id fields
    here by subclassing we add the fields: a, b, kt_prob_log2, node_prob_log2
    """

    a: int = 0                  # represents number of 0's
    b: int = 0                  # represents number of 1's
    kt_prob_log2: float = 0     # represents log2 of kt probability of current a, b
    node_prob_log2: float = 0   # represents log2 of probability of node

    def get_child(self, symbol: bool) -> CTWNode:
        """
        Return the child node corresponding to the given symbol
        """
        if not symbol:
            return self.left_child
        else:
            return self.right_child

    def get_count(self, symbol: bool) -> int:
        """
        Return the count corresponding to the given symbol
        """
        if not symbol:
            return self.a
        else:
            return self.b

    def increment_count(self, symbol: bool):
        """
        Increment the count corresponding to the given symbol
        """
        if not symbol:
            self.a += 1
        else:
            self.b += 1

    def average_log2(self, a: float, b: float) -> float:
        # return log2(0.5 * (2**a + 2**b)) using some funky math
        # TODO: magnitude of a is becoming so large (eg -1000) that 2**(a-b) becomes -inf 
        if b < a:
            temp = a-1 + log2(2**(b-a) + 1)
        else:
            temp = b-1 + log2(2**(a-b) + 1)
        return temp

    def pr_prob_log2(self, nx: int, n: int) -> float:
        """
        Compute the Laplace probability of succession for the given symbol

        Pr(nx, x) = (nx + 0.5) / (n + 1)
        where nx is the number of times symbol x has been observed
        and n is the total number of observations
        """
        return log2(2*nx + 1) - log2(2*n + 2)

    def kt_update_log2(self, next_symbol: bool):
        """
        Compute the Krichevsky Trofimov probability for the given symbol

        Pkt(a+1, b) = Pkt(a, b) * Pr(a, a+b)
        Pkt(a, b+1) = Pkt(a, b) * Pr(b, a+b)
        """
    
        # Multiply the previous kt probability by the probability of successtion for the given symbol
        nx = self.get_count(next_symbol)
        n = self.a + self.b

        self.kt_prob_log2 = self.kt_prob_log2 + self.pr_prob_log2(nx=nx, n=n)

        # If this is a leaf node, the node probability is just the kt probability
        if self.left_child == None and self.right_child == None:
            self.node_prob_log2 = self.kt_prob_log2
        # Otherwise, the node probability is the average of the kt probability and the node probability of its children
        # I.e. node_prob = 0.5 * (kt_prob + left_child_node_prob*right_child_node_prob)
        else:
            self.node_prob_log2 = self.average_log2(self.kt_prob_log2, self.left_child.node_prob_log2 + self.right_child.node_prob_log2)

        # Increment the count (i.e. a or b) for the given symbol
        self.increment_count(next_symbol)

    def _get_lines(self):
        """
        Override the _get_lines function to allow printout of the node id's, counts, and probabilities
        when we call print_tree() in CTWTree

        Adds the symbol counts and node probability to the node label
        Then remove them once we are done (since we are directly overwriting the node id)
        """
        original_id = self.id
        if not self.id:
            self.id = "ROOT"
        else:
            self.id = self.id.to01()
        self.id = self.id + ", a=" + str(self.a) + ", b=" + str(self.b) + ", node_prob_log2=" + str(self.node_prob_log2)[:5] + " (" + str(2**self.node_prob_log2)[:5] + ")"
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node

def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    test_node = CTWNode()

    # Check starting values
    alphabet = [0, 1]
    for symbol in alphabet:
        assert test_node.get_child(symbol) == None
        assert test_node.get_count(symbol) == 0

    # Check increment_count functions as expected
    test_node.increment_count(0)
    assert test_node.get_count(0) == 1
    test_node.increment_count(1)
    assert test_node.get_count(1) == 1

    # Check pr_prob_log2 returns the expected values for the following combination of values
    # Pr(a, a+b) = (a+0.5)/(a+b+1)
    test_values = [(0, 0, 1/2), (0, 1, 1/4), (2, 1, 5/8), (3, 3, 1/2), (3, 4, 7/16)]
    for a, b, result in test_values:
        np.testing.assert_almost_equal(
            2**test_node.pr_prob_log2(a, a+b),
            result
        )
    
    # Check kt_update_log2 updates the node probability to the expected value for the given sequence of symbols
    # This is a leaf node so the node probability should equal the KT probability
    test_leaf_node = CTWNode()
    test_values = [0, 1, 0, 1]
    for symbol in test_values:
        test_leaf_node.kt_update_log2(symbol)

    np.testing.assert_almost_equal(
        2**test_leaf_node.node_prob_log2,
        2**test_leaf_node.kt_prob_log2
    )
    np.testing.assert_almost_equal(
        2**test_leaf_node.node_prob_log2,
        3/128
    )

    # Check kt_update_log2 updates the node probability to the expected value for the given sequence of symbols
    # This is an internal node so the node probability should equal
    # the average of the KT probability and the product of its children's node probabilities
    left_child = CTWNode()
    right_child = CTWNode()
    left_child.node_prob_log2 = log2(5/16)
    right_child.node_prob_log2 = log2(3/8)

    test_internal_node = CTWNode(left_child=left_child, right_child=right_child)

    test_values = [0, 1, 0, 1]
    for symbol in test_values:
        test_internal_node.kt_update_log2(symbol)

    np.testing.assert_almost_equal(
        2**test_internal_node.node_prob_log2,
        0.5 * (2**test_internal_node.kt_prob_log2 +
               (2**test_internal_node.left_child.node_prob_log2) * (2**test_internal_node.right_child.node_prob_log2))
    )
    np.testing.assert_almost_equal(
        2**test_internal_node.node_prob_log2,
        9/128
    )