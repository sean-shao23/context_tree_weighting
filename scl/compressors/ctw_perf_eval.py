from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from scl.compressors.ctw_eval import gen_kth_order_markov_seq
from scl.compressors.ctw_model import CTWModel
from scl.compressors.probability_models import AdaptiveOrderKFreqModel
from scl.utils.bitarray_utils import BitArray

from math import log2
import matplotlib.pyplot as plt
import numpy as np
import time

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

    plt.savefig('time_vs_tree_size.png')

def test_rate_vs_low_input_length():
    sizes_to_test = [10, 20, 50, 100, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    aec_params = AECParams()
    k=5

    expected_rates = []
    expected_markov_rates = []
    for input_size in sizes_to_test:
        input_seq = gen_kth_order_markov_seq(k, input_size, 0.1, seed=10)
        context_str = ''.join([str(s) for s in input_seq[:k]]) 
        ctw_enc = CTWModel(k, BitArray(context_str))

        markov_enc = AdaptiveOrderKFreqModel([0, 1], k, aec_params.MAX_ALLOWED_TOTAL_FREQ)
        for symbol in input_seq[k:]:
            ctw_enc.update_model(symbol)
        for symbol in input_seq:
            markov_enc.update_model(symbol)
        
        freq_dist = ctw_enc.freqs_current.freq_list
        total_freq = ctw_enc.freqs_current.total_freq

        prob_zero = freq_dist[0]/total_freq
        prob_one = freq_dist[1]/total_freq
        entropy = prob_zero * log2(1/prob_zero) + prob_one * log2(1/prob_one)

        expected_rates.append(entropy)

        freq_dist = markov_enc.freqs_current.freq_list
        total_freq = markov_enc.freqs_current.total_freq

        prob_zero = freq_dist[0]/total_freq
        prob_one = freq_dist[1]/total_freq
        entropy = prob_zero * log2(1/prob_zero) + prob_one * log2(1/prob_one)

        expected_markov_rates.append(entropy)
    
    expected_rates_precomputed = [0.8269742450605091, 0.9955667309249244, 0.9997767050210538, 0.9180362122259074, 0.6822652758820407, 0.7869708703125857, 0.6274171294508483, 0.5066517281194405, 0.45897601415639944, 0.4887769955837905, 0.4503158486964044, 0.46338383719226595, 0.47876229644656954]
    expected_markov_rates_precomputed = [1.0, 1.0, 0.9182958340544893, 0.8112781244591328, 0.7219280948873623, 0.8112781244591328, 0.6500224216483541, 0.5293608652873645, 0.46702359828750206, 0.49261866596516846, 0.4524718376329855, 0.46427216122498605, 0.4791512835630931]

    print(expected_rates_precomputed)
    print(expected_markov_rates_precomputed)

    plt.figure()
    plt.plot(sizes_to_test, expected_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line
    plt.plot(sizes_to_test, expected_markov_rates_precomputed, 'o-')  # 'o-' means that the points will be marked and connected by a line

    plt.xlabel('Input Length (symbols)')
    plt.ylabel('Entropy (bits/symbol)')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.legend(["CTW", str(k) + "th Order Markov"])

    # TODO: Make this a dashed line?
    plt.axhline(0.1*log2(1/0.1) + 0.9*log2(1/0.9), color='green', linestyle='--')

    plt.savefig('rate_vs_input_low_length.png')
