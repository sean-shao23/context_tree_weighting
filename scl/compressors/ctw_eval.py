from scl.compressors.ctw_model import CTWModel, compress_sequence
from scl.utils.bitarray_utils import uint_to_bitarray
import time

def test_ctw_eval():
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