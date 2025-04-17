# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import uuid
import dataclasses

from bb_decoding.noise_model import NoiseModel
from bb_decoding.logical_simulation import simulate_decoding
from bb_decoding.database_utils import load_decoder_data, save_simulation_data

n_shots = 1000
# Number of Monte Carlo trials (shots) in decoding simulation.
b_readout_flip = False
s_decoder_data_id = "919e4a1fe07d4944bf6b9d7c1b523fb2"
s_name = "no leaked init, no corr, medium errors"
b_init_leaked = False
delta = .43 if b_readout_flip else .0
# Set an initial (approximate) steady-state population of leaked qubits, using a
# phenomenological constant delta (see below).

# Medium error rates
s_noise_model_filename = "./noise_yaml/simulation_noise_model.yaml"

# load decoder data and noise model from files
decoder_data = load_decoder_data(s_decoder_data_id)
noise_model = NoiseModel.from_file(s_noise_model_filename)
leaked_population = .0
if noise_model.is_leaky and b_init_leaked:
    leaked_population = .5 * (1 - 2 * delta) * noise_model.leak / noise_model.seep

simulation_input = {
    "b_readout_flip": b_readout_flip,
    "n_shots": n_shots,
    "rand_seed": 43,
    "bp_method": "ms",
    "bp_max_iterations": 200,
    # the maximum number of iterations for BP
    "osd_method": "osd_cs",
    # the OSD method. Choose from "osd_e", "osd_cs", "osd0".
    "osd_order": 7,
    # the osd search depth
    "ms_scaling_factor": 0,
    # min-sum scaling factor. If 0, a variable-scaling factor method is used
    "b_initial_state": True,
    # Start from a random initial state and a first ideal cycle.
    "leaked_population": leaked_population,
}

simulation_results, simulation_summary = simulate_decoding(
    simulation_input, decoder_data, noise_model
)
db_line = {
    "code_name": decoder_data["code_name"],
    "n_cycles": decoder_data["n_cycles"],
    "fnc": decoder_data["fnc"],
    "b_readout_flip": b_readout_flip,
    "unique_id": uuid.uuid4().hex,
    "name": s_name,
    "decoder_id": decoder_data["unique_id"],
}
db_line.update(dataclasses.asdict(noise_model))
db_line.update(simulation_summary)
simulation_input.pop("b_readout_flip")
db_line.update(simulation_input)
db_line["decoder_noise_model"] = dataclasses.asdict(decoder_data["noise_model"])

save_simulation_data(db_line, simulation_results)
