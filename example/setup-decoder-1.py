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
from bb_decoding.database_utils import save_decoder_data
from bb_decoding.decoder_data_setup import generate_decoder_data

s_noise_model_filename = "./noise_yaml/decoder_noise_model.yaml"
decoder_input = {
    "code_name": "72.12.6",  # One of "72.12.6", "144.12.12", "288.12.18", "784.24.24"
    "n_cycles": 6,  # Number of idle syndrome cycle repetitions to decode.
    "fnc": 2,  # final number of ideal cycles added (for decoding validation). 2 is enough.
    "unique_id": uuid.uuid4().hex,
    "name": "low cnot error",  # Informative name or description of the model
}

noise_model = NoiseModel.from_file(s_noise_model_filename)
decoder_data = generate_decoder_data(decoder_input, noise_model)

db_line = decoder_input.copy()
db_line.update(dataclasses.asdict(noise_model))
save_decoder_data(db_line, decoder_data)

print("Done")
