from scl.core.data_block import DataBlock
from scl.compressors.lz77 import LZ77Encoder
from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from scl.compressors.ctw_eval import gen_kth_order_markov_seq
from scl.compressors.context_tree_weighting import CTWModel, CTWModelUnicode
from scl.compressors.probability_models import AdaptiveOrderKFreqModel
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray

from gc import get_referents
from math import log2
import matplotlib.pyplot as plt
import numpy as np
import requests
import sys
import time
from types import ModuleType, FunctionType
from typing import Any

def compute_optimal_rate(enc: Any, sequence: list):
    bits = 0
    for symbol in sequence:
        freq = enc.freqs_current.frequency(symbol)
        total_freq = enc.freqs_current.total_freq
        bits += log2(total_freq/freq)
        enc.update_model(symbol)
    return bits

def test_time_vs_input_size():
    sizes_to_test = [1, 2, 3, 4, 5, 10, 20]

    times_taken = [3905.0087928771973, 6431.334733963013, 9527.854681015015, 12986.912488937378, 17629.652738571167, 35804.90827560425, 71199.5599269867]
    # times_taken = []
    if not times_taken:
        for input_size in sizes_to_test:
            input_seq = np.zeros(input_size*100000, dtype=int)
            ctw_enc = CTWModel(3, BitArray("110"))

            start_time = time.time()

            for symbol in input_seq:
                ctw_enc.update_model(symbol)

            times_taken.append((time.time() - start_time)*1000)

    print(times_taken)
    
    plt.figure()
    plt.plot(sizes_to_test, [t/1000 for t in times_taken], 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols) x 100,000')
    plt.ylabel('Encode Time (s)')
    plt.xlim([0, 21])
    plt.ylim([0, 60])
    plt.title("Encode Time vs Input Length")

    plt.savefig('time_vs_input_size.png')

def test_time_vs_tree_size():
    datasize = 100000
    sizes_to_test = [1, 2, 3, 4, 5, 10, 20]

    times_taken = [2461.505889892578, 3035.9439849853516, 3503.7267208099365, 3989.502429962158, 4312.785387039185, 6371.909141540527, 12429.552555084229]
    # times_taken = []
    if not times_taken:
        for input_size in sizes_to_test:
            input_seq = np.zeros(datasize, dtype=int)
            ctw_enc = CTWModel(input_size, BitArray("0"*input_size))

            start_time = time.time()

            for symbol in input_seq:
                ctw_enc.update_model(symbol)

            times_taken.append((time.time() - start_time)*1000)

    print(times_taken)
    
    plt.figure()
    plt.plot(sizes_to_test, [t/1000 for t in times_taken], 'o-')  # 'o-' means that the points will be marked and connected by a line

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

    # tree_sizes = [getsize(CTWTree(size, BitArray("0"*size)))/1000 for size in sizes]
    tree_sizes = [1.024, 1.52, 2.452, 4.316, 8.044, 15.5, 30.412, 60.236, 119.884, 240.716, 480.332]
    print(tree_sizes)

    plt.figure()
    plt.plot(sizes, tree_sizes, 'o-')
    plt.xlabel('Tree Size (Height)')
    plt.ylabel('Size in Memory (kB)')
    plt.title("Memory Usage vs Tree Size")

    plt.savefig('memory_vs_tree_size.png')


def test_rate_vs_input_length_markov():
    sizes_to_test = [10, 20, 50, 100, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    aec_params = AECParams()
    prob_flip = 0.1
    k=5

    expected_rates = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        input_seq = gen_kth_order_markov_seq(k, input_size, prob_flip, seed=1)
        context_str = ''.join([str(s) for s in input_seq[:5]]) 

        ctw_enc = CTWModel(5, BitArray(context_str))
        markov_enc = AdaptiveOrderKFreqModel([0, 1], 5, aec_params.MAX_ALLOWED_TOTAL_FREQ)

        ctw_bits = compute_optimal_rate(ctw_enc, input_seq[5:]) + 5
        markov_bits = compute_optimal_rate(markov_enc, input_seq)
        expected_rates.append(ctw_bits/input_size)
        expected_markov_rates.append(markov_bits/input_size)

    print(expected_rates)
    print(expected_markov_rates)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_markov_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.gca().set_ylim(bottom=0)
    plt.xscale('log')
    plt.title("Optimal Rate vs Input Length for 5th Order Markov Source with Bern(0.1)")
    plt.legend(["CTW - Depth 5", "5th Order Adaptive Model"])

    plt.axhline(prob_flip*log2(1/prob_flip) + (1-prob_flip)*log2(1/(1-prob_flip)), color='green', linestyle='--')

    plt.savefig('rate_vs_input_length.png')

test_rate_vs_input_length_markov()

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
        
    print(expected_rates)
    print(expected_markov_rates)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_markov_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.title("Optimal Rate vs Input Length for 2nd Order Tree Source with Bern(0.1)")
    plt.legend(["CTW - Depth 2", "2nd Order Adaptive Model"])

    plt.axhline(prob_flip*log2(1/prob_flip) + (1-prob_flip)*log2(1/(1-prob_flip)), color='green', linestyle='--')

    plt.savefig('rate_vs_input_length_tree.png')

def test_rate_vs_input_length_english():
    NUM_TREES = 8
    sizes_to_test = [20, 40, 60, 80, 100, 120, 150]
    sample_text = "Welcome to EE 274, a class on data compression at Stanford! This is the second offering of this course at Stanford. The reviews for the last offering are available via Stanford Carta."
    aec_params = AECParams()
    k_chars=2

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

    expected_lz77_rates = []
    for input_size in sizes_to_test:
        input_seq = sample_text[:input_size]
        input_seq_int = []
        for char in input_seq:
            input_seq_int.append(ord(char))

        lz77_enc = LZ77Encoder(initial_window=None)

        lz77_bits = lz77_enc.encode_block(DataBlock(input_seq_int))

        expected_lz77_rates.append(len(lz77_bits)/len(input_seq_int))


    print(expected_rates)
    print(expected_rates_unicode)
    print(expected_markov_rates)

    print(expected_lz77_rates)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_rates_unicode, 'o-')
    plt.plot(sizes_to_test, expected_markov_rates, 'o-')
    plt.plot(sizes_to_test, expected_lz77_rates, 'o-')


    plt.xlabel('Input Length (characters)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.legend(["CTW - Depth 16", "CTW with 8 trees - Depth 16", "16th Order Adaptive Model", "LZ77"])
    plt.ylim(5, 12)
    # plt.xlim(0, 160)

    plt.title("Optimal Rate vs Input Length for English Source")

def test_rate_vs_input_length_sherlock():
    NUM_TREES = 8

    def download_url(url, as_text=True):
        response = requests.get(url)
        if not as_text:
            return response.content
        else:
            return response.text

    sherlock = download_url("https://www.gutenberg.org/cache/epub/2852/pg2852.txt")

    print(len(sherlock))
    print(sherlock[10])
    sizes_to_test = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 381166]

    aec_params = AECParams()
    k_chars=2

    """
    expected_rates = []
    expected_rates_unicode = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        print("now testing block size", input_size)
        input_seq = sherlock[:input_size]
        input_seq_binary = []
        for i in range(len(input_seq)):
            ascii_val = ord(input_seq[i])
            if ascii_val > 255:
            if ascii_val == 8220:
                rpl = "\""
            elif ascii_val == 8212:
                rpl = "-"
            elif ascii_val == 8216:
                rpl = "'"
            elif ascii_val == 8217:
                rpl = "'"
            elif ascii_val == 8221:
                rpl = "\""
            else:
                rpl = " "

            input_seq = input_seq[:i] + rpl + input_seq[i+1:]
            input_seq_binary += uint_to_bitarray(ord(input_seq[i]), bit_width=NUM_TREES).tolist()


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


    expected_lz77_rates = []
    for input_size in sizes_to_test:
        input_seq = sherlock[:input_size]
        input_seq_int = []
        for char in input_seq:
            ascii_val = ord(char)
            if ascii_val > 255:
                if ascii_val == 8220:
                    rpl = "\""
                elif ascii_val == 8212:
                    rpl = "-"
                elif ascii_val == 8216:
                    rpl = "'"
                elif ascii_val == 8217:
                    rpl = "'"
                elif ascii_val == 8221:
                    rpl = "\""
                else:
                    rpl = " "
                ascii_val = ord(rpl)

            input_seq_int.append(ascii_val)

        lz77_enc = LZ77Encoder(initial_window=None)

        lz77_bits = lz77_enc.encode_block(DataBlock(input_seq_int))

        expected_lz77_rates.append(len(lz77_bits)/len(input_seq_int))
    """
    expected_rates = [6.9732524433176035, 5.971592792433135, 5.257272828904487, 4.700438114751906, 4.482458406434033, 3.8494508532737934, 3.573546182684467, 3.2839852364967244, 2.9872647484232635, 2.9168099448421114, 2.8715035041638837, 2.8940615606240954]
    expected_rates_unicode = [6.267817427736341, 5.73063387079042, 5.5, 5.239381020083248, 5.139366720841896, 4.710098400582242, 4.524399907656927, 4.313246786631371, 4.086808158524748, 4.037815926513225, 4.024810954938053, 4.049867378752702]
    expected_markov_rates = [7.894640123951783, 7.513910591749405, 7.313889198968683, 6.987825525754506, 6.87077492495205, 6.305347536451872, 5.803849117325835, 5.194598703095534, 4.364054588114752, 3.924154116727289, 3.536499296180615, 3.296019537811402]
    expected_lz77_rates = [9.39, 8.03, 5.526, 4.768, 4.333, 3.8372, 3.5698, 3.27895, 2.98854, 2.82899, 2.7394, 2.6592823074460994]

    print(expected_rates)
    print(expected_rates_unicode)
    print(expected_markov_rates)

    print(expected_lz77_rates)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_rates_unicode, 'o-')
    plt.plot(sizes_to_test, expected_markov_rates, 'o-')
    plt.plot(sizes_to_test, expected_lz77_rates, 'o-')


    plt.xlabel('Input Length (characters)')
    plt.ylabel('Optimal Rate (bits/symbol)')
    plt.legend(["CTW - Depth 16", "CTW with 8 trees - Depth 16", "16th Order Adaptive Model", "LZ77"])
    # plt.ylim(5.5, 8.5)
    # plt.xlim(0, 160)
    plt.xscale('log')

    plt.title("Optimal Rate vs Input Length for English Source")
