from __future__ import annotations
from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import try_lossless_compression
from scl.utils.tree_utils import BinaryNode
import copy
import numpy as np
import sys
import time
from typing import Callable

def convert_float_prob_to_int(p: float, M: int=2**16) -> int:
    """
    Convert a float probability to an integer probability
    :param p: float probability
    :param M: multiplier
    :return: integer probability
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return int(p * M)

def product_of_two_prob(p1_as_ints: (int, int), p2_as_ints: (int, int)) -> (int, int):
    p1_num = p1_as_ints[0]
    p1_denom_log = p1_as_ints[1]
    p2_num = p2_as_ints[0]
    p2_denom_log = p2_as_ints[1]


    num = p1_num * p2_num
    denom_log = p1_denom_log + p2_denom_log

    assert num != 0
    while num%2 == 0:
        num //= 2
        denom_log -= 1

    assert round(num) == num
    assert round(denom_log) == denom_log
    return (num, denom_log)

def average_of_two_prob(p1_as_ints: (int, int), p2_as_ints: (int, int)) -> (int, int):
    p1_num = p1_as_ints[0]
    p1_denom_log = p1_as_ints[1]
    p2_num = p2_as_ints[0]
    p2_denom_log = p2_as_ints[1]

    if p1_denom_log > p2_denom_log:
        p2_num *= 2**(p1_denom_log - p2_denom_log)
        num = p1_num + p2_num
        denom_log = p1_denom_log
    else:
        p1_num *= 2**(p2_denom_log - p1_denom_log)
        num = p1_num + p2_num
        denom_log = p2_denom_log

    assert num != 0
    while num%2 == 0:
        num //= 2
        denom_log -= 1

    assert round(num) == num
    assert round(denom_log+1) == denom_log+1

    # Add 1 to denom_log in order to divide by 2
    return (num, denom_log + 1)

def extract_pow_two(num: int) -> (int, int):
    max_pow = int(np.ceil(np.log2(num)))
    cur_pow = max_pow
    min_pow = 0
    while max_pow > min_pow + 1:
        cur_factor = 2**cur_pow
        if (num // cur_factor) * cur_factor == num:
            min_pow = cur_pow
        else:
            max_pow = cur_pow
        cur_pow = int(0.5 * (min_pow + max_pow))
    
    num //= 2**min_pow
    while num%2 == 0:
        num //= 2
        min_pow += 1

    return (num, min_pow)


class CTWNode(BinaryNode):
    """represents a node of the CTW tree

    NOTE: BinaryNode class already has left_child, right_child, id fields
    here by subclassing we add the fields: a, b, node_prob
    """

    # TODO: Floats are bad. Rather than saving probabilities as floats, save them as ratio of two ints
    # Denominator should always be a power of 2, so denominator can be stored as log2 of the value
    # TODO: If we store numerator as log2 as well, we can do log math (subtract instead of divide)
    a: int = 0                      # represents number of 0's
    b: int = 0                      # represents number of 1's
    kt_prob_as_ints: (int, int) = (1, 0)        # represents numerator and log of denominator for kt probability of current a, b
    node_prob_as_ints: (int, int) = (1, 0)      # represents numerator and log of denominator for node probability of current a, b

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

    def pr_prob(self, nx: int, n: int) -> float:
        num = int(2 * (nx + 0.5))
        denom = 2 * (n + 1)
        gcd = np.gcd(num, denom)
        num //= gcd
        denom //= gcd
        return num, denom

    def kt_update(self, next_symbol: bool):
        nx = self.get_count(next_symbol)
        n = self.a + self.b

        pr_num, pr_denom = self.pr_prob(nx=nx, n=n)
        extra_factor, pr_denom_log = extract_pow_two(pr_denom)

        if sys.maxsize // pr_num < self.kt_prob_as_ints[0]:
            factor = int(np.ceil(np.log2(pr_num)))
            self.kt_prob_as_ints = (self.kt_prob_as_ints[0] // (2**factor), \
                                    self.kt_prob_as_ints[1] - factor)

        self.kt_prob_as_ints = (round(self.kt_prob_as_ints[0] * pr_num / extra_factor), \
                                self.kt_prob_as_ints[1] + pr_denom_log)

        if self.left_child == None and self.right_child == None:
            self.node_prob_as_ints = self.kt_prob_as_ints
        else:
            self.node_prob_as_ints = average_of_two_prob(self.kt_prob_as_ints,
                                                product_of_two_prob(self.left_child.node_prob_as_ints,
                                                                    self.right_child.node_prob_as_ints))

        self.increment_count(next_symbol)

    def _get_lines(self):
        original_id = self.id
        if not self.id:
            self.id = "ROOT"
        self.id = str(self.id) + ", a=" + str(self.a) + ", b=" + str(self.b) + \
                  ", node_prob=" + str(self.node_prob_as_ints[0] / (2**self.node_prob_as_ints[1]))
        lines, root_node = super()._get_lines()
        self.id = original_id
        return lines, root_node

class CTWTree():

    root: CTWNode = None
    current_context: BitArray = None
    snapshot: list = None
    get_snapshot: bool = None

    def __init__(self, tree_height: int, past_context: BitArray):
        assert len(past_context) == tree_height

        self.current_context = past_context
        self.root = self.gen_tree(depth=tree_height, node_context=BitArray())

    def print_tree(self):
        self.root.print_node()

    def gen_tree(self, depth: int, node_context: BitArray) -> CTWNode:
        if depth == 0:
            return CTWNode(id=node_context, left_child=None, right_child=None)
        left_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray("0"))
        right_child = self.gen_tree(depth=depth-1, node_context=node_context + BitArray("1"))
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
            node.kt_prob_as_ints = prev_state.kt_prob_as_ints
            node.node_prob_as_ints = prev_state.node_prob_as_ints
        self.snapshot = []

    def get_root_prob(self) -> float:
        num, denom_log = self.root.node_prob_as_ints
        return num / (2**denom_log)

    # TODO: This will return 0 probability...that's bad...is that due to precision issues? Or just the nature of the tree?
    def get_symbol_prob(self, symbol: bool) -> float:
        self.snapshot = []
        self.get_snapshot = True

        prev_tree_prob_num, prev_tree_prob_denom_log = self.root.node_prob_as_ints

        self._update_node(node=self.root, context=self.current_context, symbol=symbol)

        new_tree_prob_num, new_tree_prob_denom_log = self.root.node_prob_as_ints

        denom_log = new_tree_prob_denom_log - prev_tree_prob_denom_log
        symbol_prob = (new_tree_prob_num / prev_tree_prob_num) / (2**denom_log)
        
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
            node.kt_update(symbol)
            return
        self._update_node(node=node.get_child(context[-1]), context=context[:-1], symbol=symbol)
        if self.get_snapshot:
            self.snapshot.append((node, copy.deepcopy(node)))
        node.kt_update(symbol)

class CTWModel:
    
    freqs_current: float = None
    ctw_tree: CTWTree = None

    def __init__(self, tree_height: int, context: BitArray):
        self.freqs_current = Frequencies({0: 1, 1: 1})
        self.ctw_tree = CTWTree(tree_height=tree_height, past_context=context)

    # TODO: update_model only handles 0 or 1 (properly)
    # If intended, add checks (asserts)
    # If not intended, fix
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

def test_average_of_two_prob():
    p1 = (1, 0)
    p2 = (1, 0)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        1,
    )

    p1 = (1, 2)
    p2 = (3, 5)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        0.171875,
    )

    p1 = (3, 5)
    p2 = (9, 5)
    num, denom_log = average_of_two_prob(p1, p2)
    np.testing.assert_almost_equal(
        num / (2**denom_log),
        0.1875,
    )

def test_extract_pow_two():
    val, base = extract_pow_two(1024)
    assert val == 1
    assert base == 10

    val, base = extract_pow_two(96)
    assert val == 3
    assert base == 5


def test_ctw_node():
    # TODO: Add logic to test the behavior of CTWNode
    test_node = CTWNode()

def test_ctw_tree_generation():
    test_tree = CTWTree(tree_height=3, past_context=BitArray("110"))
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        1,
    )

    test_tree.update_tree(BitArray("0100110"))
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
        7/2048,
    )
    
    test_tree.update_tree_symbol(0)
    np.testing.assert_almost_equal(
        test_tree.get_root_prob(),
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
        test_tree.get_root_prob(),
        7/2048,
    )
    np.testing.assert_almost_equal(
        test_tree.get_symbol_prob(1),
        (71/65536)/(7/2048)
    )

# TODO: Look at tests in Arithmetic Coder and copy them over
# Can borrow lossless_entropy_coder_test and lossless_test_against_bitrate functions

# TODO: Add test with English source (encode alphabet with ASCII)

# TODO: Add way to "undo" the revert of the tree (so we don't recompute tree we just computed)?
# Or make a non-overwriting update function for the nodes?
def test_ctw_model():
    def gen_input_seq(next_val_func: Callable, context: list=[1, 1, 0]) -> list:
        starting_len = len(context)
        for _ in range(DATA_SIZE):
            context.append(next_val_func(context))
        return context[starting_len:]

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

    def compress_english(sequence: list):
        unicode_sequence = [uint_to_bitarray(ord(char)) for char in sequence]
        bit_sequence = []
        for b_array in unicode_sequence:
            for bit in b_array:
                bit_sequence.append(bit)
        data_block = DataBlock(bit_sequence)
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

        return encode_len / len(sequence)

    # TODO: For DATA_SIZE any larger than ~1000 (e.g. 2000), we get 0 probability
    # Presumably because of floating point precision issues
    DATA_SIZE = 10000
    np.random.seed(0)
    time_taken = []

    start_time = time.time()

    # Source with 0th order entropy 1
    avg_codelen = compress_sequence(gen_input_seq(lambda _: np.random.binomial(1, 0.5)))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 1, decimal=1)

    start_time = time.time()

    # Source with 1st order entropy 0
    avg_codelen = compress_sequence(gen_input_seq(lambda seq: seq[-1] ^ 1))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 0, decimal=1)

    start_time = time.time()

    # Source with 2nd order entropy 0
    avg_codelen = compress_sequence(gen_input_seq(lambda seq: seq[-1] ^ seq[-2]))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 0, decimal=1)

    start_time = time.time()

    # Source with 7th order entropy 0
    # But since tree size is 3, 3rd order entropy 1
    avg_codelen = compress_sequence(gen_input_seq(next_val_func=lambda seq: seq[-1] ^ seq[-7], context=[1, 0, 0, 1, 1, 1, 0]))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 1, decimal=1)

    print("Average time (ms) per bit:", 1000*sum(time_taken)/len(time_taken)/DATA_SIZE)

    input_string = "Hello World!"
    start_time = time.time()

    bits_per_char = compress_english(input_string)

    total_time = time.time() - start_time
    print("Average time (ms) to send", len(input_string), "characters:", 1000*total_time)

    # TODO: placeholder assert (add better test)
    assert bits_per_char <= 10