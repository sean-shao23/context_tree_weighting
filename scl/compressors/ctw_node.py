from __future__ import annotations
from math import log2
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
        # return np.log2(0.5 * (2**a + 2**b)) using some funky math
        # TODO: magnitude of a is becoming so large (eg -1000) that 2**(a-b) becomes -inf 
        if b < a:
            temp =  a-1 + log2(2**(b-a) + 1)
            if 2**(b-a) < 0.001:
                pass
                # temp = a-1 + 2**(b-a)
                # print("avg log2:", a, b, 2**(a-b), 2**(b-a), temp)
                # print(self.kt_prob_log2, self.left_child.node_prob_log2 + self.right_child.node_prob_log2)
        else:
            temp =  b-1 + log2(2**(a-b) + 1)
            if 2**(a-b) < 0.001:
                pass
                # temp = b-1 + 2**(a-b)
                # print("avg log2:", a, b, 2**(a-b), 2**(b-a), temp)
                # print(self.kt_prob_log2, self.left_child.node_prob_log2, self.right_child.node_prob_log2)
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
        """
        original_id = self.id
        if not self.id:
            self.id = "ROOT"
        self.id = str(self.id) + ", a=" + str(self.a) + ", b=" + str(self.b) + ", node_prob=" + str(2**self.node_prob_log2)
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node

def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    test_node = CTWNode()
