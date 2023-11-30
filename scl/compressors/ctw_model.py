from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.compressors.ctw_tree import CTWTree
from scl.core.prob_dist import Frequencies
from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import try_lossless_compression
import copy
import numpy as np
import time
from typing import Callable

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

# TODO: Add test with English source (encode alphabet with ASCII)

# TODO: Add way to "undo" the revert of the tree (so we don't recompute tree we just computed)?
# Or make a non-overwriting update function for the nodes?
def test_ctw_model():
    # TODO: For DATA_SIZE any larger than ~1000 (e.g. 2000), we get 0 probability
    # Presumably because of floating point precision issues
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