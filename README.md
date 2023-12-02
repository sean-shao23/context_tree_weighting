# EE 274 Project Milestone: Context Tree Weighting
### Sean Shao and Caterina Zampa
## Introduction: 
A fundamental principle of compression is that prediction is compression. More precisely, a good prediction algorithm creates a good compression algorithm, and vice versa. Many of the data sources we wish to compress, from biological sequences to human language, can be effectively predicted based on the data previously observed.

Markov chains are one way to describe the probability of a given state (symbol) based on the previous states (context). A kth-order Markov chain has the property that the probability of a symbol depends only on the k immediately previous symbols. Though most data sources do not only depend on the k previous symbols, Markov chains can make a good approximation of the actual distribution. As k grows larger, the Markov model generally becomes better at predicting (and as a result compressing) because it has more information to base its prediction on. However, large values of k result in the model being slower at adapting. This is not an issue for theoretical infinite data sequences, but for real-life, finite data sources, we will see poor performance if k is too large relative to the input length. 

Context tree weighting (CTW) mixes the predictions of many Markov models of differing lengths in order to improve the adaptability of our predictor. This results in good performance for all source sequence lengths, not just infinitely long sequences. In fact, CTW achieves the theoretical lower bound for how close the performance can be to the actual source input, given the distribution of the source input is unknown. Additionally, the CTW algorithm has linear complexity, making its performance very desirable both in terms of compression and complexity for real-world applications.

## Existing Literature/Code: 

To learn about context tree weighting, we first explored its stub on Wikipedia [1], which offered a three-sentence description of what CTW is, and some links to more information about CTW. Then, we consulted the original paper by Willems, Shtarkov, and Tjalkens [2], but the language and notation used made it difficult to understand. Instead, we found lecture notes from both Stanford University [3] and the Australian National University [4] which helped us better understand the CTW algorithm and how it could be used to make predictions. 

In addition to these papers, we also referenced the GitHub repository containing Chandak, Fumin, and Zouzias’s implementation of the CTW algorithm in Go [5], though our implementation differed somewhat from theirs. We primarily used it as a sanity check for our results. 

Finally, we based the evaluation datasets on what we have used in class, what was used in the Go implementation, and a paper by Goyal, Tatwawadi, Chandak, and Ochoa [6]. Though that paper was not related to CTW, it listed 16 datasets, from real datasets of text, genomic, and audio data, to synthetic datasets based on Markov sequences. 

## Implementation and Evaluation: 
Our goal for this project, after understanding the context tree weighting method and its theoretical properties, is to implement the CTW algorithm in the Stanford Compression Library (SCL) [7]. Like the rest of the compression algorithms implemented in SCL, our goal is not to have the most optimal performance, but just to provide an easy-to-understand implementation of the algorithm. However, we still need to do some optimization, such as working around precision issues for our probability values, in order for the implementation to function as expected for reasonably large datasets and tree sizes. 

Since the Stanford Compression Library already contains an implementation of an entropy coder in the form of an adaptive arithmetic encoder/decoder, our project focuses solely on the CTW model. In order to evaluate the performance of our implementation, we will compare the performance of our CTW compressor with other existing compressors such as GZip, arithmetic coding with a kth order Markov model, and the aforementioned Go implementation [7] on a variety of datasets containing both real and synthetic data, as mentioned previously.

## Progress Report: 
Our implementation of context tree weighting currently resides in a fork of the main Stanford Compression Library repository [8]. At the time of this report, our implementation is functionally complete, being able to losslessly compress binary sequences. We also observe expected codelengths for kth order Markov sources, specifically codelength 0 for k less than or equal to tree depth, and 1 for k greater than tree depth). 
However, even though we fixed some precision and overflow issues by storing probabilities as two separate values (one for the numerator and one for the denominator), the methods used to resolve those issues are somewhat convoluted, and may not be the most efficient. As one of our primary goals for implementing the CTW algorithm in SCL was to ensure it is easily understandable, so that someone can look at our implementation to learn about CTW rather than vice versa, we will need to both refactor and better document our code to make it better understandable. 

Additionally, we may still have some minor bugs in our implementation. For example, many values are currently being stored as floating point values when they should be integers. While we are not too concerned about efficiency, for large data or tree sizes our implementation can consume a significant amount of time and memory, so we are hoping to improve the performance to some degree. A major optimization we could do is both only storing sections of the CTW tree that we need (pruning untouched branches completely), and storing the updated state of the tree after checking the probability of a given symbol (so if we do observe that symbol next, we don’t have to recompute the probability values), but we may skip these due to the limited timeframe for our project.
We are still on track with the planned timeline from our project proposal, but we don’t have much overhead to further improve our implementation. The main task we have left is to evaluate the performance of our CTW implementation on varying datasets and against various existing compressors, as described previously. 

### Timeline:
11/27 (today): Submit Project Milestone Report \
12/01: Complete Performance Evaluation of Code \
12/03: Complete First Draft of Presentation \
12/05: Revise and Practice Presentation \
12/06: Final Presentation \
12/08: Polish and Document Codebase \
12/11: Complete First Draft of Report \
12/13: Revise Report and Check with Mentor \
12/15: Submit Final Report \

## Extensions of this Project:
An extension of this project for future groups to consider could be implementing Volf’s decomposed CTW algorithm [9], which extends the CTW algorithm to handle larger alphabets, as the current implementation is only for binary alphabets. 

Another possible extension could be replacing the Krichevsky-Trofimov (KT) estimator with an adaptive or windowed KT estimator to deal with non-stationary or piecewise sequences, respectively. Alternatively, the KT estimator could be replaced entirely with a different iid estimator. 

## References:
[1] “Context Tree Weighting”,  Wikipedia, Wikimedia Foundation, Revisited last on November 27, 2023, https://en.wikipedia.org/wiki/Context_tree_weighting \
[2] F. M. J. Willems, Y. M. Shtarkov and T. J. Tjalkens, “The context-tree weighting method: basic properties,” in IEEE Transactions on Information Theory, vol. 41, no. 3, pp. 653-664, May 1995, doi: 10.1109/18.382012. Available: https://ieeexplore.ieee.org/document/382012 \
[3] Pernuter, Haim, “Context Tree Weighting (CTW)”, EE477 Universal Schemes in Information Theory, December 2011, Stanford University, https://web.stanford.edu/class/ee477/lectures2011/lecture4.pdf \
[4] Sunehag, Peter and Hutter, Marcus, “Context Tree Weighting”, 2013, The Australian National University, http://www.hutter1.net/rsise/2017/slides-ctw.pdf \
[5] Fumin, Zouzias, Anastasios, and Chandak, Shubham, “Context Tree Weighting in Go”, https://github.com/fumin/ctw \
[6] Goyal, Mohit et al., “DZip: improved neural network based general-purpose lossless compression”, Arxiv, vol 14., no. 8, August 2015, Cornell University, https://arxiv.org/pdf/1911.03572.pdf \
[7] Tatwawadi, Kedar, “Stanford Compression Library”, https://github.com/kedartatwawadi/stanford_compression_library \
[8] Shao, Sean, “Context Tree Weighting”, https://github.com/sean-shao23/context_tree_weighting \
[9] Volf, P. A. J., “Weighting Techniques in Data Compression: Theory and Algorithms,” December 2002. Available: https://www.sps.tue.nl/wp-content/uploads/2015/09/Volf2002PhDthesis.pdf \
