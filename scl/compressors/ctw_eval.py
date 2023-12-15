from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.compressors.ctw_model import CTWModel, compress_sequence
from scl.compressors.huffman_coder import (
    HuffmanEncoder,
    HuffmanDecoder,
)
from scl.compressors.probability_models import (
    AdaptiveIIDFreqModel,
    AdaptiveOrderKFreqModel,
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
import matplotlib.pyplot as plt
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


def test_ctw_model():
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
    time_taken = time.time() - start_time
    print("coding took", time_taken*1000, "(ms)")
    print(avg_codelen, 0.3*log2(1/0.3)+0.7*log2(1/0.7))


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

    # define AEC params
    aec_params = AECParams()
    freq_model_enc = AdaptiveOrderKFreqModel([0, 1], 3, aec_params.MAX_ALLOWED_TOTAL_FREQ)
    freq_model_dec = copy.deepcopy(freq_model_enc)

    # create encoder/decoder
    encoder = ArithmeticEncoder(aec_params, freq_model_enc)
    decoder = ArithmeticDecoder(aec_params, freq_model_dec)

    start_time = time.time()
    is_lossless, output_len, _  = try_lossless_compression(DataBlock(markov_seq), encoder, decoder)
    time_taken = time.time() - start_time
    print("adaptive coding took", time_taken*1000, "(ms)")
    avg_codelen = output_len / DATA_SIZE
    print(f"avg_codelen: {avg_codelen:.3f}")



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

def test_and_plot():
    sizes = [1000, 2000, 4000, 6000, 8000, 10000]
    ctw_e = []
    ctw_d = []
    ctw_r = []
    adapt_e = []
    adapt_d = []
    adapt_r = []
    huff_e = []
    huff_d = []
    huff_r = []
    lz_e = []
    lz_d = []
    lz_r = []
    aec_params = AECParams()
    for data_size in sizes:
        markov_seq = gen_kth_order_markov_seq(3, data_size, prob_bit_flip=0.1)
        seq_as_datablock = DataBlock(markov_seq)
        prob_dist = seq_as_datablock.get_empirical_distribution(order=0)
        
        freq_model_enc = CTWModel(3, BitArray("000"))
        freq_model_dec = copy.deepcopy(freq_model_enc)
        encoder = ArithmeticEncoder(aec_params, freq_model_enc)
        decoder = ArithmeticDecoder(aec_params, freq_model_dec)
        is_lossless, output_len, _, enc_time, dec_time = try_lossless_compression(seq_as_datablock, encoder, decoder)
        assert is_lossless
        ctw_e.append(enc_time*1000)
        ctw_d.append(dec_time*1000)
        ctw_r.append(output_len / data_size)

        freq_model_enc = AdaptiveOrderKFreqModel([0, 1], 3, aec_params.MAX_ALLOWED_TOTAL_FREQ)
        freq_model_dec = copy.deepcopy(freq_model_enc)
        encoder = ArithmeticEncoder(aec_params, freq_model_enc)
        decoder = ArithmeticDecoder(aec_params, freq_model_dec)
        is_lossless, output_len, _, enc_time, dec_time = try_lossless_compression(seq_as_datablock, encoder, decoder)
        assert is_lossless
        adapt_e.append(enc_time*1000)
        adapt_d.append(dec_time*1000)
        adapt_r.append(output_len / data_size)

        encoder = HuffmanEncoder(prob_dist)
        decoder = HuffmanDecoder(prob_dist)
        is_lossless, output_len, _, enc_time, dec_time = try_lossless_compression(seq_as_datablock, encoder, decoder)
        assert is_lossless
        huff_e.append(enc_time*1000)
        huff_d.append(dec_time*1000)
        huff_r.append(output_len / data_size)

        encoder = LZ77Encoder(initial_window=None)
        decoder = LZ77Decoder(initial_window=None)
        is_lossless, output_len, _, enc_time, dec_time = try_lossless_compression(seq_as_datablock, encoder, decoder)
        assert is_lossless
        lz_e.append(enc_time*1000)
        lz_d.append(dec_time*1000)
        lz_r.append(output_len / data_size)

    plt.figure()
    plt.plot(sizes, ctw_e, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes, adapt_e, 'o-')
    plt.plot(sizes, huff_e, 'o-')
    plt.plot(sizes, lz_e, 'o-')


    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Time (ms))')
    plt.legend(["CTW - Depth 3", "3rd Order Adaptive Model", "Huffman", "LZ77"])

    plt.title("Encode Time vs Input Length")
    plt.savefig('enc_time_vs_length_all.png')

    plt.figure()
    plt.plot(sizes, ctw_d, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes, adapt_d, 'o-')
    plt.plot(sizes, huff_d, 'o-')
    plt.plot(sizes, lz_d, 'o-')


    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Time (ms))')
    plt.legend(["CTW - Depth 3", "3rd Order Adaptive Model", "Huffman", "LZ77"])

    plt.title("Decode Time vs Input Length")
    plt.savefig('dec_time_vs_length_all.png')


    plt.figure()
    plt.plot(sizes, ctw_r, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes, adapt_r, 'o-')
    plt.plot(sizes, huff_r, 'o-')
    plt.plot(sizes, lz_r, 'o-')


    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Rate (bits/symbol))')
    plt.legend(["CTW - Depth 3", "3rd Order Markov", "Huffman", "LZ77"])
    prob_flip = 0.1
    plt.gca().set_ylim(bottom=0)
    plt.axhline(prob_flip*log2(1/prob_flip) + (1-prob_flip)*log2(1/(1-prob_flip)), color='green', linestyle='--')

    plt.title("Compression Rate vs Input Length")
    plt.savefig('rate_vs_length_all.png')

test_and_plot()