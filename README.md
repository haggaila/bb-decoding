A package allowing to execute Monte Carlo decoding simulations of bivariate-bicycle quantum error correction codes, with a state-dependent noise model going Pauli noise (including qubit decay and leakage), and conditional pre-measurement check-qubit flips suppressing faults in the readout gates.

Source code and data for the simulations presented in:  
_Feedforward suppression of readout-induced faults in quantum error correction_  
Liran Shirizly, Dekel Meirom, Malcolm Carroll, Haggai Landa,  
[arXiv:2504.13083](https://arxiv.org/abs/2504.13083)

This code is based on the sources and simulations described in:  
_High-threshold and lowoverhead fault-tolerant quantum memory_  
[Nature **627**, 778 (2024)](https://www.nature.com/articles/s41586-024-07107-7)
https://github.com/sbravyi/BivariateBicycleCodes  

Decoding relies on the software implementation of the Belief Propagation with the Ordered Statistics Decoder at:  
https://pypi.org/project/ldpc/

setup-decoder-1.py is the offline part of the decoder that constructs check matrices,
syndrome measurement circuits, and decoding matrices for a particular quantum code.
The output is saved to a local database structure with a main file in csv format and pickle decoder data files.

run-decoder-1.py is the online part of the decoder that simulates error correction circuits. The output is saved to a local database structure with a main file in csv format and pickle simulation data files.

