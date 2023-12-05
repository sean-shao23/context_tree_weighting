from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.compressors.ctw_model import CTWModel, compress_sequence
from scl.compressors.huffman_coder import (
    HuffmanEncoder,
    HuffmanDecoder,
)
from scl.compressors.probability_models import (
    AdaptiveIIDFreqModel,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, ProbabilityDist, get_avg_neg_log_prob
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import (
    lossless_entropy_coder_test,
    create_random_binary_file,
    try_file_lossless_compression,
    lossless_test_against_expected_bitrate,
)
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.compressors.lz77 import (
    LZ77Encoder,
    LZ77Decoder,
)
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
import copy
from math import log2
import numpy as np
import tempfile
import time
import os

def gen_kth_order_markov_seq(k: int, num_samples: int, prob_bit_flip: float=0.5, seed: int=0):
    """generate a kth order Markov distribution for testing.

    Defined on alphabet {0, 1}, the distribution is defined like
    X_n = X_{n-1} + X_{n-k} + Ber(prob_bit_flip) mod 2
    """
    assert num_samples >= k
    np.random.seed(seed)

    markov_samples = np.zeros(num_samples, dtype=int)
    markov_samples[0:k] = np.random.randint(0, 2, k)
    random_bits = np.random.rand(num_samples - k)
    for i in range(k, num_samples):
        markov_samples[i] = (markov_samples[i - 1] + markov_samples[i - k] + (random_bits[i - k] < prob_bit_flip)) % 2
    return markov_samples

# Look at tests in Arithmetic Coder and copy them over
# Can borrow lossless_entropy_coder_test and lossless_test_against_bitrate functions

# TODO: Tests to add:
# test with different tree depths
# results to show: rate, time taken
# test against: huffman, adaptive (k-th order markov) arithmetic coding, modern coder (e.g. gzip/bzip2)
# kth order markov -- test against 

# Graph time vs input size and/or tree depth

def test_adaptive_order_k_arithmetic_coding():
    """
    Test CTW coding on 2nd order Markov
    - Check if encoding/decodng is lossless
    - Check if the compression is close to expected for k = 0, 1, 2, 3
    - Verify that 0th order matches the adaptive IID exactly.
    """

    DATA_SIZE = 2**16

    start_time = time.time()
    markov_seq = gen_kth_order_markov_seq(3, DATA_SIZE, 0.3)
    time_taken = time.time() - start_time
    print("generating input took", time_taken, "(ms)")

    start_time = time.time()
    avg_codelen = compress_sequence(markov_seq)
    np.testing.assert_almost_equal(avg_codelen, 0.3*log2(1/0.3)+0.7*log2(1/0.7), decimal=1)
    time_taken = time.time() - start_time
    print("coding took", time_taken*1000, "(ms)")
    print(avg_codelen, 0.3*log2(1/0.3)+0.7*log2(1/0.7))



def test_huffman_encoding():
    """
    Test Huffman encoder to compare with CTW
    """

    #def test_huffman_coding_dyadic():
    """test huffman coding on dyadic distributions

    On dyadic distributions Huffman coding should be perfectly equal to entropy
    1. Randomly generate data with the given distribution
    2. Construct Huffman coder using the given distribution
    3. Encode/Decode the block
    """
    DATA_SIZE = 2**16

    markov_seq = gen_kth_order_markov_seq(3, DATA_SIZE)
    data_block_huffman = DataBlock(markov_seq)
    prob_dist = data_block_huffman.get_empirical_distribution(order=0)
    # create encoder decoder

    encoder = HuffmanEncoder(prob_dist)
    decoder = HuffmanDecoder(prob_dist)

    # perform compression
    start_time_huffman = time.time()
    is_lossless, output_len, _ = try_lossless_compression(data_block_huffman, encoder, decoder)
    time_taken_huffman = time.time() - start_time_huffman
    print("huffman coding took", time_taken_huffman*1000, "(ms)")
    avg_codelen = output_len / DATA_SIZE
    print(f"avg_codelen: {avg_codelen:.3f}")

    # get optimal codelen
    #optimal_codelen = get_avg_neg_log_prob(prob_dist, data_block)
    assert is_lossless, "Lossless compression failed"


def test_lz77_multiblock_file_encode_decode():
    """full test for LZ77Encoder and LZ77Decoder

    - create a sample file
    - encode the file using LZ77Encoder
    - perform decoding and check if the compression was lossless

    """
    DATA_SIZE = 2**16

    markov_seq = gen_kth_order_markov_seq(3, DATA_SIZE)
    data_block_lz77 = DataBlock(markov_seq)

    #initial_window = [44, 45, 46] * 5
    # define encoder, decoder
    encoder = LZ77Encoder(initial_window=markov_seq)
    decoder = LZ77Decoder(initial_window=markov_seq)

    start_time_lz77 = time.time()
    is_lossless, output_len, _ = try_lossless_compression(data_block_lz77, encoder, decoder)
    time_taken_lz77 = time.time() - start_time_lz77
    avg_codelen = output_len / DATA_SIZE
    print(f"avg_codelen: {avg_codelen:.3f}")
    print("lz77 coding took", time_taken_lz77*1000, "(ms)")


   
"""
def test_adaptive_arithmetic_coding():
    Test if AEC coding is working as expcted for different parameter settings
    - Check if encoding/decodng is lossless
    - Check if the compression is close to optimal

    NUM_SAMPLES = 1000

    DATA_SIZE = 2**16

    markov_seq = gen_kth_order_markov_seq(3, DATA_SIZE)
    data_block_lz77 = DataBlock(markov_seq)

    # trying out some random frequencies/aec_parameters
    data_freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
    ]

    params_list = [
        AECParams(),
        AECParams(),
        AECParams(DATA_BLOCK_SIZE_BITS=12),
        AECParams(DATA_BLOCK_SIZE_BITS=12, PRECISION=16),
    ]
    
    ## create adaptive coder
    for freq, params in zip(data_freqs_list, params_list):

        # define initial distribution to be uniform
        uniform_dist = Frequencies({a: 1 for a in freq.alphabet})

        # create encoder/decoder model
        # NOTE: important to make a copy, as the encoder updates the model, and we don't want to pass
        # the update model around
        freq_model_enc = AdaptiveIIDFreqModel(
            freqs_initial=uniform_dist, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ
        )
        freq_model_dec = copy.deepcopy(freq_model_enc)

        # create enc/dec
        encoder = ArithmeticEncoder(params, freq_model_enc)
        decoder = ArithmeticDecoder(params, freq_model_dec)
        def lossless_entropy_coder_test_new(
            encoder: DataEncoder,
            decoder: DataDecoder,
            freq: Frequencies,
            data_size: int,
            encoding_optimality_precision: bool = None,
            seed: int = 0,
        ):
            Checks if the given entropy coder performs lossless compression and optionally if it is
            "optimal".

            NOTE: the notion of optimality is w.r.t to the avg_log_probability of the randomly
            generated input.
            Example usage is for compressors such as Huffman, AEC, rANS etc.

            Args:
                encoder (DataEncoder): Encoder to test with
                decoder (DataDecoder): Decoder to test lossless compression with
                freq (Frequencies): freq distribution used to generate random i.i.d data
                data_size (int): the size of the data to generate
                encoding_optimality_precision (bool, optional): Optionally (if not None) check if the average log_prob is close to the avg_codelen. Defaults to None.
                seed (int, optional): _description_. seed to generate random data. Defaults to 0.
            # generate random data
            prob_dist = freq.get_prob_dist()
            data_block = get_random_data_block(prob_dist, data_size, seed=seed)
            avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)

            # check if encoding/decoding is lossless
            start_time_adaptive = time.time()
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )
            time_taken_adaptive = time.time() - start_time_adaptive
            print("adaptive coding took", time_taken_adaptive*1000, "(ms)")

            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = (encode_len) / data_block.size
            print(f" avg_log_prob={avg_log_prob:.3f}, avg_codelen: {avg_codelen:.3f}")

            # check whether arithmetic coding results are close to optimal codelen
            if encoding_optimality_precision is not None:
                err_msg = f"avg_codelen={avg_codelen} is not {encoding_optimality_precision} close to avg_log_prob={avg_log_prob}"
                assert np.abs(avg_codelen - avg_log_prob) < encoding_optimality_precision, err_msg

            assert is_lossless

        lossless_entropy_coder_test_new(
            encoder, decoder, freq, NUM_SAMPLES, encoding_optimality_precision=1e-1, seed=0
        )
        
        """
