A package allowing to execute Monte Carlo decoding simulations of bivariate-bicycle quantum error correction codes, with a state-dependent noise model going Pauli noise (including qubit decay and leakage).

This code is based on the sources at https://github.com/sbravyi/BivariateBicycleCodes

setup-decoder-1.py is the offline part of the decoder that constructs check matrices,
syndrome measurement circuits, and decoding matrices for a particular quantum code.
The output is saved to a local database structure with a main file in csv format and pickle decoder data files.

run-decoder-1.py is the online part of the decoder that simulates error correction circuits.
It relies on the software implementation of the Belief Propagation with the Ordered Statistics Decoder at https://pypi.org/project/ldpc/

