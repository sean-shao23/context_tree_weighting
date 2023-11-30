from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from scl.compressors.ctw_model import CTWModel, compress_sequence
from scl.core.prob_dist import Frequencies
from scl.compressors.probability_models import (
    AdaptiveIIDFreqModel,
)
from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray
from scl.utils.test_utils import (
    lossless_entropy_coder_test,
    lossless_test_against_expected_bitrate,
)
import copy
import numpy as np
import time

# TODO: Look at tests in Arithmetic Coder and copy them over
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

    def _generate_kth_order_markov(k: int, num_samples: int, seed: int = 0):
        """generate a 2nd order Markov distribution for testing.

        Defined on alphabet {0,1,2}, the distribution is defined like
        X_n = X_{n-1} + X_{n-2} + Ber(1/2) mod 3

        The entropy rate is 1 bit/symbol.

        The stationary distribution is the uniform distribution.
        """
        assert num_samples > k
        rng = np.random.default_rng(seed)
        # TODO: change to range 0-to-1 so the probability isnt hardcoded to 0.3
        random_bits = rng.choice(10, size=num_samples - k)
        markov_samples = np.zeros(num_samples, dtype=int)
        markov_samples[0] = rng.choice(2)
        markov_samples[1] = rng.choice(2)
        for i in range(2, num_samples):
            markov_samples[i] = (markov_samples[i - 1] + markov_samples[i - k] + (random_bits[i - k] < 3)) % 2
        return markov_samples

    DATA_SIZE = 2**16

    start_time = time.time()
    markov_seq = _generate_kth_order_markov(3, DATA_SIZE)
    time_taken = time.time() - start_time
    print("generating input took", time_taken, "(ms)")

    start_time = time.time()
    avg_codelen = compress_sequence(markov_seq)
    np.testing.assert_almost_equal(avg_codelen, 0.3*np.log2(1/0.3)+0.7*np.log2(1/0.7), decimal=1)
    time_taken = time.time() - start_time
    print("coding took", time_taken*1000, "(ms)")
    print(avg_codelen, 0.3*np.log2(1/0.3)+0.7*np.log2(1/0.7))

def test_ctw_english():
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
    
    # Send "Hello World!"
    input_string = "Hello World!"
    start_time = time.time()

    bits_per_char = compress_english(input_string)

    total_time = time.time() - start_time
    print("Average time (ms) to send", len(input_string), "characters:", 1000*total_time)

    # TODO: placeholder assert (add better test)
    assert bits_per_char > 0
    print(bits_per_char)