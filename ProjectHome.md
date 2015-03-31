Implementation of neural nets that won the ICML 2013 multimodal learning challenge hosted on Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-multi-modal-learning

Requirements: NVIDIA Fermi GPUs needed: GTX 4xx and above, Titan. Matlab

C++/CUDA code for fully connected and convolutional neural networks is provided.
MEX files bindings provided. Scripts to reproduce the winning solution are also provided.

Uses conv kernels from Alex Krizhevsky's cuda-convnet project: http://code.google.com/p/cuda-convnet/

homepage: www.cs.toronto.edu/~tang

## Acknowledgements ##
Thanks to Alex Krizhevsky for providing fast convolutional kernels

Niclas Borlin for the hungarian/Munkres algorithm Matlab implementation