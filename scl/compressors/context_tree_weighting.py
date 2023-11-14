from __future__ import annotations
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.tree_utils import BinaryNode
from typing import Literal
import copy
import numpy as np

class CTWNode(BinaryNode):
    """represents a node of the CTW tree

    NOTE: BinaryNode class already has left_child, right_child, id fields
    here by subclassing we add the fields: a, b, node_prob
    """

    a: int = 0              # represents number of 0's
    b: int = 0              # represents number of 1's
    kt_prob: float = 1      # represents kt probability for current a, b
    node_prob: float = 1    # probability of node

    def get_child(self, symbol: bool) -> CTWNode:
        if not symbol:
            return self.left_child
        else:
            return self.right_child

    def get_count(self, symbol: bool) -> int:
        if not symbol:
            return self.a
        else:
            return self.b

    def increment_count(self, symbol: bool):
        if not symbol:
            self.a += 1
        else:
            self.b += 1

    def pr_prob(self, nx: int, n: int, alpha: float) -> float:
        return (nx + alpha) / (n + 1)

    def kt_update(self, next_symbol: bool, alpha: float):
        nx = self.get_count(next_symbol)
        n = self.a + self.b
        self.kt_prob = self.kt_prob * self.pr_prob(nx=nx, n=n, alpha=alpha)
        if self.left_child == None and self.right_child == None:
            self.node_prob = self.kt_prob
        else:
            self.node_prob = 0.5 * (self.kt_prob
                                    + self.left_child.node_prob * self.right_child.node_prob)
        self.increment_count(next_symbol)

    def _get_lines(self):
        original_id = self.id
        if not self.id:
            self.id = "ROOT"
        self.id += ", a=" + str(self.a) + ", b=" + str(self.b) + ", node_prob=" + str(self.node_prob)
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node

class CTWTree():

    root: CTWNode = None
    current_context: BitArray = None
    snapshot: list = None
    get_snapshot: bool = None
    alpha: float = 0.5

    def __init__(self, tree_height: int, past_context: BitArray):
        assert len(past_context) == tree_height

        self.root = self.gen_tree(depth=tree_height, node_context=BitArray())

        self.current_context = past_context

    def print_tree(self):
        self.root.print_node()

    def gen_tree(self, depth: int, node_context: BitArray) -> CTWNode:
        if depth == 0:
            return CTWNode(id=node_context, left_child=None, right_child=None)
        left_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray(0))
        right_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray(1))
        return CTWNode(id=node_context, left_child=left_child, right_child=right_child)
    
    def update_tree(self, sequence: BitArray):
        for symbol in sequence:
            self._update_tree_symbol(symbol)

    def revert_tree(self):
        for node, prev_state in self.snapshot:
            assert type(node) == CTWNode
            assert type(prev_state) == CTWNode
            node.a = prev_state.a
            node.b = prev_state.b
            node.kt_prob = prev_state.kt_prob
            node.node_prob = prev_state.node_prob
        self.snapshot = []

    def get_symbol_prob(self, symbol: bool):
        self.snapshot = []
        self.get_snapshot = True
        context_prob = self.root.node_prob
        self._update_node(node=self.root, context=self.current_context, symbol=symbol)
        symbol_context_prob = self.root.node_prob
        symbol_prob = symbol_context_prob/context_prob
        self.revert_tree()
        self.get_snapshot = False
        return symbol_prob

    def _update_tree_symbol(self, next_symbol: bool):
        self._update_node(node=self.root, context=self.current_context, symbol=next_symbol)
        self.current_context = self.current_context[1:] + uint_to_bitarray(next_symbol)

    def _update_node(self, node: CTWNode, context: str, symbol: bool):
        if len(context) == 0:
            if self.get_snapshot:
                self.snapshot.append((node, copy.deepcopy(node)))
            node.kt_update(symbol, self.alpha)
            return
        self._update_node(node=node.get_child(context[-1]), context=context[:-1], symbol=symbol)
        if self.get_snapshot:
            self.snapshot.append((node, copy.deepcopy(node)))
        node.kt_update(symbol, self.alpha)

def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    pass

def test_ctw_tree_generation():
    test = CTWTree(tree_height=3, past_context=BitArray("110"))
    np.testing.assert_almost_equal(
        test.root.node_prob,
        1,
    )

    test.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test.root.node_prob,
        7/2048,
    )

    test.update_tree(BitArray("0"))
    np.testing.assert_almost_equal(
        test.root.node_prob,
        153/65536,
    )

def test_ctw_tree_probability():
    test = CTWTree(tree_height=3, past_context=BitArray("110"))
    test.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test.get_symbol_prob(0),
        (153/65536)/(7/2048)
    )
    np.testing.assert_almost_equal(
        test.root.node_prob,
        7/2048,
    )
    np.testing.assert_almost_equal(
        test.get_symbol_prob(1),
        (71/65536)/(7/2048)
    )