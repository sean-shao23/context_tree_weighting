from __future__ import annotations
from scl.utils.tree_utils import BinaryNode
import numpy as np
import sys

# TODO: There's a lot of floats flying around when they should be ints
# Identify and fix if needed
def product_of_two_prob(p1_as_ints: (int, int), p2_as_ints: (int, int)) -> (int, int):
    """
    Multiply two probabilities together (probabilities represented by two integers)
    :param p1_as_ints: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the first probability
    :param p1_as_ints: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the second probability
    :return: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the product of the probabilities
    """
    num = p1_as_ints[0] * p2_as_ints[0]
    denom_log = p1_as_ints[1] + p2_as_ints[1]

    assert num != 0

    # Simplify numerator and denominator where possible
    # TODO: Is this simplification really needed/how much does it help?
    while num%2 == 0:
        num //= 2
        denom_log -= 1

    # Ensure results are integers
    assert round(num) == num
    assert round(denom_log) == denom_log

    return (num, denom_log)

def average_of_two_prob(p1_as_ints: (int, int), p2_as_ints: (int, int)) -> (int, int):
    """
    Get the average of two probabilities (probabilities represented by two integers)
    :param p1_as_ints: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the first probability
    :param p1_as_ints: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the second probability
    :return: Tuple(int, int)
                Represents the numerator and log2 of the denominator     
                of the average of the probabilities
    """
    p1_num = p1_as_ints[0]
    p1_denom_log = p1_as_ints[1]
    p2_num = p2_as_ints[0]
    p2_denom_log = p2_as_ints[1]

    # Convert the two probabilites to have the same denominator
    # Then sum the numerators
    if p1_denom_log > p2_denom_log:
        p2_num *= 2**(p1_denom_log - p2_denom_log)
        denom_log = p1_denom_log
    else:
        p1_num *= 2**(p2_denom_log - p1_denom_log)
        denom_log = p2_denom_log

    num = p1_num + p2_num

    assert num != 0

    # Simplify numerator and denominator where possible
    while num%2 == 0:
        num //= 2
        denom_log -= 1

    # Ensure results are integers
    assert round(num) == num
    assert round(denom_log+1) == denom_log+1

    # Add 1 to denom_log in order to divide the result by 2
    return (num, denom_log + 1)

class CTWNode(BinaryNode):
    """
    Represents a node of the CTW tree

    NOTE: BinaryNode class already has left_child, right_child, id fields
    here by subclassing we add the fields: a, b, kt_prob_as_ints, and node_prob_as_ints
    """

    a: int = 0                                  # represents number of 0's
    b: int = 0                                  # represents number of 1's
    kt_prob_as_ints: (int, int) = (1, 0)        # represents numerator and log2 of denominator for kt probability of current a, b
    node_prob_as_ints: (int, int) = (1, 0)      # represents numerator and log2 of denominator for node probability of current a, b

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

    def pr_prob(self, nx: int, n: int) -> float:
        """
        Compute the Laplace probability of succession for the given symbol

        Pr(nx, x) = (nx + 0.5) / (n + 1)
        where nx is the number of times symbol x has been observed
        and n is the total number of observations
        """
        num = int(2 * (nx + 0.5))
        denom = 2 * (n + 1)

        # Simply 
        gcd = np.gcd(num, denom)
        num //= gcd
        denom //= gcd

        return num, denom

    def kt_update(self, next_symbol: bool):
        """
        Compute the Krichevsky Trofimov probability for the given symbol

        Pkt(a+1, b) = Pkt(a, b) * Pr(a, a+b)
        Pkt(a, b+1) = Pkt(a, b) * Pr(b, a+b)
        """
    
        # Compute the probablity of succession for the given symbol
        pr_num, pr_denom = self.pr_prob(nx=self.get_count(next_symbol), n=self.a+self.b)

        # Extract all the "2's" from the denominator 
        # I.e. pr_denom = pr_denom_factor * 2^(pr_denom_log)
        pr_denom_factor = pr_denom
        pr_denom_log = 0
        while pr_denom_factor%2 == 0:
            pr_denom_factor //= 2
            pr_denom_log += 1

        # If the result of self.kt_prob_as_ints[0] * pr_num will be too large, scale down the values to fit
        if sys.maxsize // pr_num < self.kt_prob_as_ints[0]:
            # Compute factor = ceil(log2(pr_num))
            factor = 1
            while 2**factor < pr_num:
                factor += 1

            # Divide the numerator by 2^(factor)
            # And subtract log2 of the denominator by factor
            self.kt_prob_as_ints = (self.kt_prob_as_ints[0] // (2**factor), \
                                    self.kt_prob_as_ints[1] - factor)

        # Multiply the previous kt probability by the probability of successtion to get the new kt probability
        # We divide pr_denom_factor (the non-power-of-two part of the denomator) from the numerator
        # in order to keep the denominator a power of two
        # The resulting numerator may not be an integer, so round the numerator to the nearest integer

        # TODO: Could we just use integer division instead of rounding?
        self.kt_prob_as_ints = (round(self.kt_prob_as_ints[0] * pr_num / pr_denom_factor), \
                                self.kt_prob_as_ints[1] + pr_denom_log)

        # If this is a leaf node, the node probability is just the kt probability
        if self.left_child == None and self.right_child == None:
            self.node_prob_as_ints = self.kt_prob_as_ints
        # Otherwise, the node probability is the average of the kt probability and the node probability of its children
        # I.e. node_prob = 0.5 * (kt_prob + left_child_node_prob + right_child_node_prob)
        else:
            self.node_prob_as_ints = average_of_two_prob(self.kt_prob_as_ints,
                                                product_of_two_prob(self.left_child.node_prob_as_ints,
                                                                    self.right_child.node_prob_as_ints))

        # Increment the count (i.e. a or b) for the given symbol
        self.increment_count(next_symbol)

    def _get_lines(self):
        """
        Override the _get_lines function to allow printout of the node id's, counts, and probabilities
        when we call print_tree() in CTWTree

        Adds the symbol counts and node probability to the node label
        """
        original_id = self.id
        if not self.id:
            self.id = "ROOT"
        self.id = str(self.id) + ", a=" + str(self.a) + ", b=" + str(self.b) + \
                  ", node_prob=" + str(self.node_prob_as_ints[0] / (2**self.node_prob_as_ints[1]))
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node


def test_average_of_two_prob():
    # Average of 1/1 and 1/1
    p1 = (1, 0)
    p2 = (1, 0)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        1,
    )

    # Average of 1/4 and 3/32
    p1 = (1, 2)
    p2 = (3, 5)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        0.171875,
    )

    # Average of 3/32 and 9/32
    p1 = (3, 5)
    p2 = (9, 5)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        0.1875,
    )

def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    test_node = CTWNode()
