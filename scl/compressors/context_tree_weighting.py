from __future__ import annotations
from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import try_lossless_compression
from scl.utils.tree_utils import BinaryNode
import copy
import numpy as np

def convert_float_prob_to_int(p, M=1000):
    """
    Convert a float probability to an integer probability
    :param p: float probability
    :param M: multiplier
    :return: integer probability
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return max(1, int(p * M))

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
        self.id = str(self.id) + ", a=" + str(self.a) + ", b=" + str(self.b) + ", node_prob=" + str(self.node_prob)
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node

class CTWTree():

    root: CTWNode = None
    current_context: BitArray = None
    snapshot: list = None
    get_snapshot: bool = None
    alpha: float = None

    def __init__(self, tree_height: int, past_context: BitArray, alpha: float = 0.5):
        assert len(past_context) == tree_height

        self.alpha = alpha
        self.current_context = past_context
        self.root = self.gen_tree(depth=tree_height, node_context=BitArray())


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
            self.update_tree_symbol(symbol)

    def revert_tree(self):
        for node, prev_state in self.snapshot:
            assert type(node) == CTWNode
            assert type(prev_state) == CTWNode
            node.a = prev_state.a
            node.b = prev_state.b
            node.kt_prob = prev_state.kt_prob
            node.node_prob = prev_state.node_prob
        self.snapshot = []

    # TODO: This will return 0 probability...that seems bad/wrong?
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

    def update_tree_symbol(self, next_symbol: bool):
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

class CTWModel:
    
    freqs_current: float = None
    ctw_tree: CTWTree = None

    def __init__(self, tree_height: int, context: BitArray, alpha: float = 0.5):
        self.freqs_current = Frequencies({0: 1, 1: 1})
        self.ctw_tree = CTWTree(tree_height=tree_height, past_context=context, alpha=alpha)

    def update_model(self, symbol: bool):
        self.ctw_tree.update_tree_symbol(symbol)
        new_dist = {0: convert_float_prob_to_int(self.ctw_tree.get_symbol_prob(0)),
                    1: convert_float_prob_to_int(self.ctw_tree.get_symbol_prob(1))}
        self.freqs_current = Frequencies(new_dist)

        aec_params = AECParams() # params used for arithmetic coding in SCL
        assert self.freqs_current.total_freq <= aec_params.MAX_ALLOWED_TOTAL_FREQ, (
            f"Total freq {self.freqs_current.total_freq} is greater than "
            f"max allowed total freq {aec_params.MAX_ALLOWED_TOTAL_FREQ} for arithmetic coding in SCL. This leads to"
            f"precision and speed issues. Try reducing the total freq by a factor of 2 or more."
        )
        self.freqs_current._validate_freq_dist(self.freqs_current.freq_dict) # check if freqs are valid datatype


def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    test_node = CTWNode()

# TODO: Add tests with different alpha values
def test_ctw_tree_generation():
    test_tree = CTWTree(tree_height=3, past_context=BitArray("110"))
    np.testing.assert_almost_equal(
        test_tree.root.node_prob,
        1,
    )

    test_tree.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test_tree.root.node_prob,
        7/2048,
    )

    test_tree.update_tree_symbol(0)
    np.testing.assert_almost_equal(
        test_tree.root.node_prob,
        153/65536,
    )

def test_ctw_tree_probability():
    test_tree = CTWTree(tree_height=3, past_context=BitArray("110"))
    test_tree.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test_tree.get_symbol_prob(0),
        (153/65536)/(7/2048)
    )
    np.testing.assert_almost_equal(
        test_tree.root.node_prob,
        7/2048,
    )
    np.testing.assert_almost_equal(
        test_tree.get_symbol_prob(1),
        (71/65536)/(7/2048)
    )

def test_ctw_model():
    def compress_sequence(sequence: list):
        data_block = DataBlock(sequence)
        # define AEC params
        aec_params = AECParams()
        # define encoder/decoder models
        # NOTE: important to make a copy, as the encoder updates the model, and we don't want to pass
        # the update model around
        # TODO: What should the context be, if anything?
        freq_model_enc = CTWModel(3, BitArray("110"))
        freq_model_dec = copy.deepcopy(freq_model_enc)

        # create encoder/decoder
        encoder = ArithmeticEncoder(aec_params, freq_model_enc)
        decoder = ArithmeticDecoder(aec_params, freq_model_dec)

        # check if encoding/decoding is lossless
        is_lossless, encode_len, _ = try_lossless_compression(
            data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
        )

        assert is_lossless

        return encode_len / data_block.size

    # TODO: For DATA_SIZE any larger than ~1000 (e.g. 2000), we get 0 probability
    # Presumably because of floating point precision issues
    DATA_SIZE = 1000
    np.random.seed(0)

    input_seq = [1, 1, 0]
    for _ in range(DATA_SIZE):
        input_seq.append(np.random.binomial(1, 0.5))
    input_seq = input_seq[3:]

    avg_codelen = compress_sequence(input_seq)
    np.testing.assert_almost_equal(avg_codelen, 1, decimal=1)

    input_seq = [1, 1, 0]
    for i in range(DATA_SIZE-1):
        input_seq.append(input_seq[-1] ^ 1)
    input_seq = input_seq[3:]

    avg_codelen = compress_sequence(input_seq)
    np.testing.assert_almost_equal(avg_codelen, 0, decimal=1)

    input_seq = [1, 0, 0, 1, 1, 1, 0]
    for _ in range(DATA_SIZE):
        input_seq.append(input_seq[-1] ^ input_seq[-2])
    input_seq = input_seq[7:]

    avg_codelen = compress_sequence(input_seq)
    np.testing.assert_almost_equal(avg_codelen, 0, decimal=1)

    input_seq = [1, 0, 0, 1, 1, 1, 0]
    for _ in range(DATA_SIZE):
        input_seq.append(input_seq[-1] ^ input_seq[-7])
    input_seq = input_seq[7:]

    avg_codelen = compress_sequence(input_seq)
    np.testing.assert_almost_equal(avg_codelen, 1, decimal=1)
