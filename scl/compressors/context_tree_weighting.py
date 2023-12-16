from __future__ import annotations

from scl.compressors.probability_models import FreqModelBase
from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.core.prob_dist import Frequencies
from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import try_lossless_compression
from scl.utils.tree_utils import BinaryNode

from collections import deque
import copy
from math import log2
import numpy as np
import string
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

    # Checks if result will be 0
    if int(p * M) <= 0:
        print("Multiplier", M, "too small for probability", p)
        return 1

    return int(p * M)

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


# TODO: Branch pruning will reduce the memory usage of our tree (currently exponential wrt tree height)
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

    # TODO: Add way to "undo" the revert of the tree (so we actually see this symbol next, we don't have to recompute the tree again)
    def get_symbol_prob(self, symbol: bool) -> float:
        """
        Compute the probability of seeing the given symbol based on the current state of the CTW tree

        P(symbol | context) = P(symbol, context) / P(context)
        """
        assert symbol == 0 or symbol == 1

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


class CTWModel(FreqModelBase):
    """
    Represents the CTW model

    Store the CTW tree and updates the frequency distribution accordingly
    """
    
    freqs_current: float = None     # Stores the frequency distribution of the CTW tree
    ctw_tree: CTWTree = None        # Stores the CTW tree itself

    def __init__(self, tree_height: int, context: BitArray):
        """
        Initialize the CTW tree with the given height and context
        Initialize the frequency distribution
        """
        self.ctw_tree = CTWTree(tree_height=tree_height, past_context=context)

        prob_of_zero = self.ctw_tree.get_symbol_prob(0)
        default_dist = {0: convert_float_prob_to_int(prob_of_zero),
                        1: convert_float_prob_to_int(1-prob_of_zero)}
        assert default_dist[0] == default_dist[1]
        self.freqs_current = Frequencies(default_dist)

    def update_model(self, symbol: bool):
        """
        Update the model with the given symbol
        """
        assert symbol == 0 or symbol == 1

        # Update the tree with the given symbol
        self.ctw_tree.update_tree([symbol])
        
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

NUM_TREES = 8
class CTWModelUnicode(FreqModelBase):
    """
    Represents the CTW model for coding unicode

    Store eight CTW trees (one for each bit) and updates the frequency distribution accordingly
    """
    
    freqs_current: float = None     # Stores the frequency distribution based on the CTW trees
    ctw_trees: list = None        # Stores the eight CTW trees


    def __init__(self, tree_height: int, context: BitArray):
        """
        Initialize the frequency dict to uniform distribution
        and CTW tree with the given heigh and context
        """
        self.freqs_current = Frequencies({chr(i): 1 for i in range(2**NUM_TREES)})
        self.ctw_trees = [CTWTree(tree_height=tree_height, past_context=context) for _ in range(NUM_TREES)]

    def update_model(self, symbol: string):
        """
        Update the model with the given character
        """
        assert len(symbol) == 1
        unicode_value = uint_to_bitarray(ord(symbol), bit_width=8)
        # Update the tree with the given symbol
        for i in range(NUM_TREES):
            self.ctw_trees[i].update_tree_symbol(unicode_value[i])
            self.ctw_trees[i].update_context(unicode_value)
        
        # Compute the new symbol probabilities for the CTW trees
        probs_of_zero = [ctw_tree.get_symbol_prob(0) for ctw_tree in self.ctw_trees]
        new_dist = {}
        for i in range(2**NUM_TREES):
            probability = 1
            binary_i = uint_to_bitarray(i, bit_width=NUM_TREES)
            for j in range(NUM_TREES):
                if binary_i[j] == 0:
                    probability *= probs_of_zero[j]
                else:
                    probability *= 1 - probs_of_zero[j]
            new_dist[chr(i)] = convert_float_prob_to_int(probability, M=2**30)

        # Update the frequency distribution
        self.freqs_current = Frequencies(new_dist)

        aec_params = AECParams() # params used for arithmetic coding in SCL
        assert self.freqs_current.total_freq <= aec_params.MAX_ALLOWED_TOTAL_FREQ, (
            f"Total freq {self.freqs_current.total_freq} is greater than "
            f"max allowed total freq {aec_params.MAX_ALLOWED_TOTAL_FREQ} for arithmetic coding in SCL. This leads to"
            f"precision and speed issues. Try reducing the total freq by a factor of 2 or more."
        )
        self.freqs_current._validate_freq_dist(self.freqs_current.freq_dict) # check if freqs are valid


def test_ctw_node():
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

def compress_sequence(sequence: list, tree_depth: int=3, context: BitArray=BitArray("110")):
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
    freq_model_enc = CTWModel(tree_depth, context)
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

def compress_english(input_string: list):
    """
    Create an arithmetic encoder/decoder pair using the CTW model

    Send the given sequence and ensure it was transmitted losslessly

    Return the bits/symbol of the encoded result
    """
    data_block = DataBlock(input_string)
    # define AEC params
    aec_params = AECParams()
    # define encoder/decoder models
    # NOTE: important to make a copy, as the encoder updates the model, and we don't want to pass
    # the update model around
    context = "A"
    context_bitarray = BitArray()
    for char in context:
        context_bitarray += uint_to_bitarray(ord(char), bit_width=8)
    freq_model_enc = CTWModelUnicode(8, context_bitarray)
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

def test_ctw_model():
    DATA_SIZE = 2**10
    def gen_input_seq(next_val_func: Callable, context: list=[1, 1, 0]) -> list:
        """
        Create a sequence using next_val_func() for the next value of the sequence,
        starting with the given context (not included in final sequence)
        """
        starting_len = len(context)
        for _ in range(DATA_SIZE):
            context.append(next_val_func(context))
        return context[starting_len:]

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

def test_ctw_english_as_binary():
    input_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    
    # Convert the text into a binary sequence by converting each character into its unicode value
    input_string_binary = []
    for char in input_string:
        input_string_binary += uint_to_bitarray(ord(char), bit_width=8).tolist()
    
    start_time = time.time()

    # Compresses the binary sequence and confirms it was losslessly sent
    bits_per_bit = compress_sequence(input_string_binary)

    total_time = time.time() - start_time
    print("Average time (ms) per character:", 1000*total_time/len(input_string))

    # If input string is too short, the overhead of the arithmetic encoder may be high enough that bits_per_char is >= 8
    bits_per_char = bits_per_bit*8
    assert bits_per_char < 8
    print("Average bits per character:", bits_per_char)



def test_ctw_english():
    input_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

    start_time = time.time()

    # Compresses the text and confirms it was losslessly sent
    bits_per_char = compress_english(input_string)

    total_time = time.time() - start_time
    print("Average time (ms) per character:", 1000*total_time/len(input_string))

    # If input string is too short, the overhead of the arithmetic encoder may be high enough that bits_per_char is >= 8
    assert bits_per_char < 8
    print("Average bits per character:", bits_per_char)