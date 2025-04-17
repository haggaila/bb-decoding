# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# This code is based on the sources at https://github.com/sbravyi/BivariateBicycleCodes
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass


@dataclass
class NoiseModel:
    prep: float = 0.0
    """Qubit preparation (init) error probability."""

    id_s: float = 0.0
    """Short-duration (parallel to CNOT gate) idle-qubit error probability."""

    id_l: float = 0.0
    """Long-duration (parallel to readout) idle-qubit error probability."""

    cnot: float = 0.0
    """CNOT gate error probability."""

    meas: float = 0.0
    """Readout error probability."""

    bias: float = None
    """State-dependent readout bias, i.e. p(0|1) - p(1|0)."""

    flip: float = None
    """X-gate error probability."""

    leak: float = None
    """Probability of a check qubit to leak during measurement if it is in the state |1>."""

    seep: float = None
    """Probability of a leaked qubit to return to the state |1> during measurement."""

    back: float = None
    """Backaction probability of a leaked check qubit to scramble a data qubit during a CNOT."""

    corr: float = None
    """Probability of a preparation error correlated with (and following) a readout error."""

    @property
    def is_biased(self) -> bool:
        """Returns True if there are qubits with a biased (state-dependent) readout error."""
        return self.bias != 0.0

    @property
    def is_leaky(self) -> bool:
        """Returns True if there are qubits with a (state-dependent) leakage error."""
        return self.leak != 0.0

    @property
    def is_state_dependent(self) -> bool:
        """Returns True if the noise model is state dependent."""
        return self.is_biased or self.is_leaky

    @staticmethod
    def from_file(s_noise_file: str) -> NoiseModel:
        if s_noise_file == "":
            noise_model = NoiseModel()
        elif os.path.isfile(s_noise_file):
            with open(s_noise_file, "r") as fin:
                noise_file = yaml.full_load(fin)
            noise_model = NoiseModel(**noise_file)
        else:
            raise Exception(f"Noise model file {s_noise_file} not found.")
        print("Noise model loaded from ", s_noise_file)
        return noise_model
