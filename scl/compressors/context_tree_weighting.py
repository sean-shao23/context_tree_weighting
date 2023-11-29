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

# TODO: Original paper describes storage complexity as linear in D...but the size of the tree is ~ 2^D
# TODO: (related to above) do path pruning to save memory?

# TODO: Seems if DATA_SIZE (see testing section) is too large, this returns probability 0
def convert_float_prob_to_int(p: float, M: int=2**16) -> int:
    """
    Convert a float probability to an integer probability
    :param p: float probability
    :param M: multiplier
    :return: integer probability
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"

    # Checks if result will be 0
    if int(p * M) <= 0:
        print(p, p*M)
        assert False

    return int(p * M)

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

        # Update the context with the symbol we just added to the tree
        # Remove the oldest symbol from the context
        self.current_context = self.current_context[1:] + uint_to_bitarray(next_symbol, bit_width=1)

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

class CTWModel:
    """
    Represents the CTW model

    Store the CTW tree and updates the frequency distribution accordingly
    """
    
    freqs_current: float = None     # Stores the frequency distribution of the CTW tree
    ctw_tree: CTWTree = None        # Stores the CTW tree itself

    def __init__(self, tree_height: int, context: BitArray):
        """
        Initialize the frequency dict to uniform distribution
        and CTW tree with the given heigh and context
        """
        self.freqs_current = Frequencies({0: 1, 1: 1})
        self.ctw_tree = CTWTree(tree_height=tree_height, past_context=context)

    # TODO: update_model only handles 0 or 1 (properly)
    # If intended, add checks (asserts)
    # If not intended, fix
    def update_model(self, symbol: bool):
        """
        Update the model with the given symbol
        """
        # Update the tree with the given symbol
        self.ctw_tree.update_tree_symbol(symbol)
        
        # Compute the new symbol probabilities for the CTW tree
        prob_of_zero = self.ctw_tree.get_symbol_prob(0)
        new_dist = {0: convert_float_prob_to_int(prob_of_zero),
                    1: convert_float_prob_to_int(1-prob_of_zero)}

        # Update the frequency distribution
        self.freqs_current = Frequencies(new_dist)

        aec_params = AECParams() # params used for arithmetic coding in SCL
        assert self.freqs_current.total_freq <= aec_params.MAX_ALLOWED_TOTAL_FREQ, (
            f"Total freq {self.freqs_current.total_freq} is greater than "
            f"max allowed total freq {aec_params.MAX_ALLOWED_TOTAL_FREQ} for arithmetic coding in SCL. This leads to"
            f"precision and speed issues. Try reducing the total freq by a factor of 2 or more."
        )
        self.freqs_current._validate_freq_dist(self.freqs_current.freq_dict) # check if freqs are valid

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

# TODO: Look at tests in Arithmetic Coder and copy them over
# Can borrow lossless_entropy_coder_test and lossless_test_against_bitrate functions

# TODO: Add test with English source (encode alphabet with ASCII)

# TODO: Add way to "undo" the revert of the tree (so we don't recompute tree we just computed)?
# Or make a non-overwriting update function for the nodes?
def test_ctw_model():
    def gen_input_seq(next_val_func: Callable, context: list=[1, 1, 0]) -> list:
        """
        Create a sequence using next_val_func() for the next value of the sequence,
        starting with the given context (not included in final sequence)
        """
        starting_len = len(context)
        for _ in range(DATA_SIZE):
            context.append(next_val_func(context))
        return context[starting_len:]

    def compress_sequence(sequence: list):
        """
        Create an arithmetic encoder/decoder pair using the CTW model

        Send the given sequence and ensure it was transmitted losslessly

        Return the bits/symbol of the encoded result
        """
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

    # TODO: Compress with 8 seperate trees (one for each bit) rather than 1 tree?
    def compress_english(sequence: list):
        """
        Send a string of English characters as a list of binary values

        Return the bits/character of the encoded result
        """
        unicode_sequence = [uint_to_bitarray(ord(char), bit_width=8) for char in sequence]
        bit_sequence = []
        for b_array in unicode_sequence:
            for bit in b_array:
                bit_sequence.append(bit)
        
        avg_codelen = compress_sequence(bit_sequence)

        return avg_codelen * len(bit_sequence) / len(sequence)

    # TODO: For DATA_SIZE any larger than ~1000 (e.g. 2000), we get 0 probability
    # Presumably because of floating point precision issues
    DATA_SIZE = 2**10
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
    avg_codelen = compress_sequence(gen_input_seq(lambda seq: seq[-1] ^ seq[-3]))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 0, decimal=1)

    start_time = time.time()

    # Source with 7th order entropy 0
    # But since tree size is 3, 3rd order entropy 1
    avg_codelen = compress_sequence(gen_input_seq(next_val_func=lambda seq: seq[-1] ^ seq[-7], context=[1, 0, 0, 1, 1, 1, 0]))

    time_taken.append(time.time() - start_time)
    np.testing.assert_almost_equal(avg_codelen, 1, decimal=1)

    print("Average time (ms) per bit:", 1000*sum(time_taken)/len(time_taken)/DATA_SIZE)

    # Send "Hello World!"
    input_string = "Hello World!"
    start_time = time.time()

    bits_per_char = compress_english(input_string)

    total_time = time.time() - start_time
    print("Average time (ms) to send", len(input_string), "characters:", 1000*total_time)

    # TODO: placeholder assert (add better test)
    assert bits_per_char > 0
    print(bits_per_char)