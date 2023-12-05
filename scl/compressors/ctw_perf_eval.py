from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from scl.compressors.ctw_eval import gen_kth_order_markov_seq
from scl.compressors.ctw_model import CTWModel, CTWModelUnicode
from scl.compressors.ctw_tree import CTWTree
from scl.compressors.probability_models import AdaptiveOrderKFreqModel
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray

from gc import get_referents
from math import log2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from types import ModuleType, FunctionType
from typing import Any

def compute_optimal_rate(enc: Any, sequence: list):
    bits = 0
    for symbol in sequence:
        enc.update_model(symbol)
        freq_dist = enc.freqs_current.freq_list
        total_freq = enc.freqs_current.total_freq

        entropy = 0
        for freq in freq_dist:
            prob = freq/total_freq
            entropy += prob * log2(1/prob)

        bits += entropy
    return bits

def test_time_vs_input_size():
    sizes_to_test = [1, 2, 3, 4, 5, 10, 20]

    """
    times_taken = []
    for input_size in sizes_to_test:
        input_seq = np.zeros(input_size*100000, dtype=int)
        ctw_enc = CTWModel(3, BitArray("110"))

        start_time = time.time()

        for symbol in input_seq:
            ctw_enc.update_model(symbol)

        times_taken.append((time.time() - start_time)*1000)
    """

    times_taken_precomputed = [2908.867597579956, 5795.073986053467, 8563.17138671875, 11593.276977539062, 14253.265142440796, 28525.282859802246, 57110.273122787476]
    print(times_taken_precomputed)
    
    plt.figure()
    plt.plot(sizes_to_test, [t/1000 for t in times_taken_precomputed], 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols) x 100,000')
    plt.ylabel('Encode Time (s)')
    plt.xlim([0, 21])
    plt.ylim([0, 60])
    plt.title("Encode Time vs Input Length")

    plt.savefig('time_vs_input_size.png')

def test_time_vs_tree_size():
    datasize = 100000
    sizes_to_test = [1, 2, 3, 4, 5, 10, 20]

    """
    times_taken = []
    for input_size in sizes_to_test:
        input_seq = np.zeros(datasize, dtype=int)
        ctw_enc = CTWModel(input_size, BitArray("0"*input_size))

        start_time = time.time()

        for symbol in input_seq:
            ctw_enc.update_model(symbol)

        times_taken.append((time.time() - start_time)*1000)
    """

    times_taken_precomputed = [2012.902021408081, 2474.2307662963867, 2937.960147857666, 3301.0332584381104, 3753.5312175750732, 5839.991807937622, 10435.01329421997]
    print(times_taken_precomputed)
    
    plt.figure()
    plt.plot(sizes_to_test, [t/1000 for t in times_taken_precomputed], 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Tree Size (Height)')
    plt.ylabel('Encode Time (s)')
    plt.xlim([0, 21])
    plt.ylim([0, 12])
    plt.title("Encode Time vs Tree Size")

    plt.savefig('time_vs_tree_size.png')

def test_memory_vs_tree_size():
    BLACKLIST = type, ModuleType, FunctionType

    def getsize(obj):
        """sum size of object & members."""
        if isinstance(obj, BLACKLIST):
            raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
        seen_ids = set()
        size = 0
        objects = [obj]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size

    sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tree_sizes = [getsize(CTWTree(size, BitArray("0"*size)))/1000 for size in sizes]

    plt.figure()
    plt.plot(sizes, tree_sizes)
    plt.xlabel('Tree Size (Height)')
    plt.ylabel('Size in Memory (kB)')
    plt.title("Memory Usage vs Tree Size")

    plt.savefig('memory_vs_tree_size.png')


def test_rate_vs_input_length_markov():
    sizes_to_test = [10, 20, 50, 100, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    aec_params = AECParams()
    prob_flip = 0.1
    k=5

    """
    expected_rates = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        input_seq = gen_kth_order_markov_seq(k, input_size, prob_flip, seed=1)
        context_str = ''.join([str(s) for s in input_seq[:k]]) 

        ctw_enc = CTWModel(k, BitArray(context_str))
        markov_enc = AdaptiveOrderKFreqModel([0, 1], k, aec_params.MAX_ALLOWED_TOTAL_FREQ)

        ctw_bits = compute_optimal_rate(ctw_enc, input_seq[k:]) + k
        markov_bits = compute_optimal_rate(markov_enc, input_seq)
        expected_rates.append(ctw_bits/input_size)
        expected_markov_rates.append(markov_bits/input_size)
    """

    expected_rates_precomputed = [0.9764031350542443, 0.9660057524677498, 0.8832024532985352, 0.920452269316836, 0.7726705314975587, 0.7054213821198038, 0.6269632874770548, 0.568649166831337, 0.5179458987400668, 0.4976660642383781, 0.485771863954453, 0.4765432680362103, 0.47327161233011217]
    expected_markov_rates_precomputed = [1.0, 0.9525313543649847, 0.918772449626383, 0.9099632824307652, 0.7962795901223189, 0.7433608348425556, 0.6717780794227087, 0.6049086038043991, 0.5410284098012595, 0.5125271659470453, 0.49491314213474624, 0.4811054209226618, 0.4758971526515169]

    print(expected_rates_precomputed)
    print(expected_markov_rates_precomputed)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_markov_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.title("Optimal Rate vs Input Length for 5th Order Markov Source with Bern(0.1)")
    plt.legend(["CTW", "5th Order Markov Model"])

    # TODO: Make this a dashed line?
    plt.axhline(prob_flip*log2(1/prob_flip) + (1-prob_flip)*log2(1/(1-prob_flip)), color='green', linestyle='--')

    plt.savefig('rate_vs_input_length.png')

def test_rate_vs_input_length_tree_source():
    def gen_tree_source(num_samples, prob_bit_flip):
        np.random.seed(2)
        k=2

        markov_samples = np.zeros(num_samples, dtype=int)
        markov_samples[0:k] = np.random.randint(0, 2, k)
        random_bits = np.random.rand(num_samples - k)
        for i in range(k, num_samples):
            if markov_samples[i - 1] == 1:
                markov_samples[i] = 0 + (random_bits[i - k] < prob_bit_flip)
            else:
                markov_samples[i] = (markov_samples[i - 1] + markov_samples[i - k] + (random_bits[i - k] < prob_bit_flip)) % 2
        return markov_samples

    sizes_to_test = [10, 20, 50, 100, 300, 500, 1000, 2000, 5000, 10000]
    aec_params = AECParams()
    prob_flip = 0.1
    k=2

    expected_rates = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        input_seq = gen_tree_source(input_size, prob_flip)
        context_str = ''.join([str(s) for s in input_seq[:k]]) 

        ctw_enc = CTWModel(k, BitArray(context_str))
        markov_enc = AdaptiveOrderKFreqModel([0, 1], k, aec_params.MAX_ALLOWED_TOTAL_FREQ)

        ctw_bits = compute_optimal_rate(ctw_enc, input_seq[k:]) + k
        markov_bits = compute_optimal_rate(markov_enc, input_seq)
        expected_rates.append(ctw_bits/input_size)
        expected_markov_rates.append(markov_bits/input_size)
        
    expected_rates_precomputed = expected_rates
    expected_markov_rates_precomputed = expected_markov_rates

    print(expected_rates_precomputed)
    print(expected_markov_rates_precomputed)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_markov_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.title("Optimal Rate vs Input Length for 2nd Order Tree Source with Bern(0.1)")
    plt.legend(["CTW", "2nd Order Markov"])

    # TODO: Make this a dashed line?
    plt.axhline(prob_flip*log2(1/prob_flip) + (1-prob_flip)*log2(1/(1-prob_flip)), color='green', linestyle='--')

    plt.savefig('rate_vs_input_length_tree.png')

def test_rate_vs_input_length_english():
    NUM_TREES = 8


    sizes_to_test = [20, 40, 60, 80, 100, 120, 150]
    sample_text = "Welcome to EE 274, a class on data compression at Stanford! This is the second offering of this course at Stanford. The reviews for the last offering are available via Stanford Carta."
    aec_params = AECParams()
    k_chars=2


    """
    expected_rates = []
    expected_rates_unicode = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        input_seq = sample_text[:input_size]
        input_seq_binary = []
        for char in input_seq:
            input_seq_binary += uint_to_bitarray(ord(char), bit_width=NUM_TREES).tolist()


        context_bitarray = BitArray(input_seq_binary[:k_chars*NUM_TREES])
   
        ctw_enc = CTWModel(k_chars*NUM_TREES, context_bitarray)
        ctw_unicode_enc = CTWModelUnicode(k_chars*NUM_TREES, context_bitarray)
        markov_enc = AdaptiveOrderKFreqModel([chr(i) for i in range(256)], k_chars, aec_params.MAX_ALLOWED_TOTAL_FREQ)


        ctw_bits = compute_optimal_rate(ctw_enc, input_seq_binary[k_chars*NUM_TREES:]) + k_chars*NUM_TREES
        ctw_unicode_bits = compute_optimal_rate(ctw_unicode_enc, input_seq[k_chars:]) + k_chars*NUM_TREES
        markov_bits = compute_optimal_rate(markov_enc, input_seq)


        expected_rates.append(ctw_bits/input_size)
        expected_rates_unicode.append(ctw_unicode_bits/input_size)
        expected_markov_rates.append(markov_bits/input_size)
    """


    expected_rates_precomputed = [7.767702521503873, 7.591960185250805, 7.333576247710156, 7.121935185837094, 6.918347069483133, 6.741904222534488, 6.511946794167895]
    expected_rates_unicode_precomputed = [6.451259916162104, 6.268055797216225, 6.125702512671365, 6.065929549807966, 6.0090381323044415, 5.954248159267637, 5.895249096437932]
    expected_markov_rates_precomputed = [8.0, 7.999784244802657, 7.999640408004429, 7.999361937720744, 7.999068198644859, 7.998812208526192, 7.998421190295794]
    print(expected_rates_precomputed)
    print(expected_rates_unicode_precomputed)
    print(expected_markov_rates_precomputed)


    plt.figure()
    plt.plot(sizes_to_test, expected_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_rates_unicode_precomputed, 'o-')
    plt.plot(sizes_to_test, expected_markov_rates_precomputed, 'o-')


    plt.xlabel('Input Length (characters)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.legend(["CTW", "CTW with 8 trees", str(k_chars) + "nd Order Markov"])
    # plt.ylim(5.5, 8.5)
    plt.xlim(0, 160)

    plt.title("Optimal Rate vs Input Length for English Source")
    plt.savefig('rate_vs_input_length_english.png')
