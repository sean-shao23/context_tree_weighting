# Context Tree Weighting
## Sean Shao and Caterina Zampa

This repository contains a full functional implementation of the Context Tree Weighting (CTW) compression algorithm, coded in Python 3.
CTW mixes the predictions of many Markov models of differing lengths in order to improve the adaptability of the predictor. This results in good performance for all source sequence lengths, not just infinitely long sequences.
We implemented CTW in the Stanford Compression Library (SCL), which can be found at the following link: https://github.com/kedartatwawadi/stanford_compression_library

To setup SCL, follow the instructions in the README of SCL: https://github.com/kedartatwawadi/stanford_compression_library?tab=readme-ov-file#getting-started

Because the goal of SCL is “not meant to be fast or efficient implementation, but rather for educational purpose,” our implementation is focused on being as straightforward to understand as possible. As a result, our code is not as optimized as other implementations, and only works on binary alphabets, but it still serves its purpose of both demonstrating how the context tree weighting method works and how its performance compares to other compression techniques. 

The three core classes of our implementation are CTWNode, CTWTree, and CTWModel. CTWModel uses the other two classes, as well as the existing implementation of arithmetic coding in SCL, to allow an input to be encoded or decoded. Further information on these classes can be found in the Final Report, attached below. 

As expected, performance evaluations on our algorithm show a linear relationship between the conpression time and both the length of the input source and the tree size, and an exponential relationship between the memory footprint and the depth of the tree. Results also show little discrepancy in performance for the CTW compressor compared to a kth-order Adaptive Arithmetic Encoder for an exact order Markov source, while seeing a notable speedup by CTW when the input source distribution does not exactly match. Additional results yielded by our performance evaluations can be found in the Final Report at the provided link below.  

Future work for this project includes implementing branch pruning, which would reduce the static memory usage of our model. Additionally, since our algorithm is only intended to process binary alphabets, an additional extension to this project could be implementing the decomposed CTW algorithm to allow larger alphabets. Finally, another possible extension could be replacing the Krichevsky-Trofimov (KT) estimator with an adaptive or windowed KT estimator to deal with non-stationary or piecewise sequences, respectively. Alternatively, the KT estimator could be replaced with an entirely different iid estimator.

#### Link to Final Report: https://docs.google.com/document/d/1us2u3a0XirPm2olOTUo2SawydYO8NIY298SC8sohmnE
#### Link to Presentation: https://docs.google.com/presentation/d/1OBl4hdcz0h3uKht-4pY9Dg5KSpwTNX_O-P-hVNoShI8
