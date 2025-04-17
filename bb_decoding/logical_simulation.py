# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# This code is based on the sources at https://github.com/sbravyi/BivariateBicycleCodes
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np
from typing import Dict
from datetime import datetime
from ldpc import bposd_decoder

from bb_decoding.noise_model import NoiseModel
from bb_decoding.circuit_simulation import (
    generate_noisy_circuit,
    get_detector_history,
    simulate_errors,
)


def simulate_decoding(
    simulation_input: Dict, decoder_data: Dict, noise_model: NoiseModel
) -> (Dict, Dict):
    n_shots = simulation_input["n_shots"]
    # Number of Monte Carlo trials (shots) in the simulation.
    b_readout_flip = simulation_input["b_readout_flip"]
    # Whether to simulate the readout flip protocol
    leaked_population = simulation_input["leaked_population"]
    # The overall mean leaked population to assign in check qubits, for entering the decoding
    # phase with the qubits close to a leakage/seepage steady state.
    rand_seed = simulation_input.get("rand_seed", 42)
    output_step = simulation_input.get("output_step", 50)
    # Number of shots to skip between console output lines if no error occurred

    # BP-OSD decoder parameters
    bp_method = simulation_input["bp_method"]
    bp_max_iterations = simulation_input["bp_max_iterations"]  # BP iterations cutoff
    osd_method = simulation_input["osd_method"]  # "osd_e", "osd_cs", "osd0"
    osd_order = simulation_input["osd_order"]  # The osd search depth
    ms_scaling_factor = simulation_input["ms_scaling_factor"]
    # min-sum scaling factor. If 0, a variable-scaling factor method is used
    b_initial_state = simulation_input["b_initial_state"]
    # Start from a random initial state and a first ideal cycle.

    HX_decoder = decoder_data["HX_decoder"]
    HZ_decoder = decoder_data["HZ_decoder"]
    x_probs = decoder_data["x_probs"]
    z_probs = decoder_data["z_probs"]
    lin_order = decoder_data["lin_order"]
    data_qubits = decoder_data["data_qubits"]
    data_qubit_indices = decoder_data["data_qubit_indices"]
    x_checks = decoder_data["x_checks"]
    z_checks = decoder_data["z_checks"]
    cycle = decoder_data["cycle"]
    HX = decoder_data["HX"]
    HZ = decoder_data["HZ"]
    lx = decoder_data["lx"]
    lz = decoder_data["lz"]
    z_logical_row = decoder_data["z_logical_row"]
    x_logical_row = decoder_data["x_logical_row"]
    ell = decoder_data["ell"]
    m = decoder_data["m"]
    n = decoder_data["n"]
    k = decoder_data["k"]
    n_cycles = decoder_data["n_cycles"]
    fnc = decoder_data["fnc"]
    n2 = m * ell
    cycle_repeated = n_cycles * cycle

    # begin decoding
    x_bpd = bposd_decoder(
        HX_decoder,  # the parity check matrix
        channel_probs=x_probs,
        max_iter=bp_max_iterations,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        osd_method=osd_method,
        osd_order=osd_order,
    )
    z_bpd = bposd_decoder(
        HZ_decoder,  # the parity check matrix
        channel_probs=z_probs,
        max_iter=bp_max_iterations,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        osd_method="osd_cs",
        osd_order=osd_order,
    )

    b_state_sim = noise_model.is_state_dependent
    rng = np.random.default_rng(rand_seed)
    bad_shots = 0
    bad_shots_z = 0
    bad_shots_x = 0
    t1 = datetime.now()
    bp_converged = np.full((n_shots, 2), 0)
    bp_iterations = np.full((n_shots, 2), 0)
    unsatisfied_fraction = np.full((n_shots, 2), 0.0)
    # The fraction of check detectors (the result of the xor in time of each check)
    # that are "unsatisfied" (not 0), averaged over all cycles. This is not currently
    # saved to the results and should be removed.

    mean_ground = np.full((n2 * n_cycles, 2), 0.0)
    mean_leaked = np.full((n2 * n_cycles, 2), 0.0)
    for i_shot in range(n_shots):
        circ = generate_noisy_circuit(cycle_repeated, noise_model, rng)

        z_initial_state = np.zeros(2 * n, dtype=int)
        x_initial_state = np.zeros(2 * n, dtype=int)
        if b_initial_state:
            z_initial_state[data_qubit_indices] = rng.binomial(1, 0.5, n)
            x_initial_state[data_qubit_indices] = rng.binomial(1, 0.5, n)
            full_circuit = cycle.copy()
            if b_state_sim:
                full_circuit += [("ON",)]
            full_circuit += circ
        else:
            full_circuit = circ.copy()
        if b_state_sim:
            full_circuit += [("OFF",)]
        for i in range(fnc):
            full_circuit += cycle

        (
            z_pre_readout_history,
            z_syndrome_history,
            z_syndrome_map,
            z_state,
            x_pre_readout_history,
            x_syndrome_history,
            x_syndrome_map,
            x_state,
        ) = simulate_errors(
            full_circuit,
            n2,
            lin_order,
            rng,
            z_initial_state,
            x_initial_state,
            noise_model,
            b_readout_flip,
            leaked_population,
        )
        assert len(z_pre_readout_history) == n_cycles * n2
        assert len(x_pre_readout_history) == n_cycles * n2

        z_detector_history, z_unsatisfied_fraction = get_detector_history(
            z_syndrome_history,
            z_syndrome_map,
            x_checks,
            n2,
            n_cycles,
            fnc,
            b_initial_state,
        )
        assert HZ_decoder.shape[0] == len(z_detector_history)
        z_bpd.decode(z_detector_history)
        bp_converged[i_shot, 0] = z_bpd.converge
        bp_iterations[i_shot, 0] = z_bpd.iter
        unsatisfied_fraction[i_shot, 0] = z_unsatisfied_fraction
        mean_ground[:, 0] += np.asarray(z_pre_readout_history == 0, dtype=float)
        mean_leaked[:, 0] += np.asarray(z_pre_readout_history == 3, dtype=float)

        z_low_weight_error = z_bpd.osdw_decoding
        assert len(z_low_weight_error) == HZ.shape[1]
        z_syndrome_history_augmented_guessed = (HZ @ z_low_weight_error) % 2
        z_syndrome_final_logical_guessed = z_syndrome_history_augmented_guessed[
            z_logical_row : (z_logical_row + k)
        ]
        if b_initial_state:
            z_state = (z_state + z_initial_state) % 2
        z_state_data_qubits = [z_state[lin_order[q]] for q in data_qubits]
        z_syndrome_final_logical = (lx @ z_state_data_qubits) % 2
        b_z_success = np.array_equal(
            z_syndrome_final_logical_guessed, z_syndrome_final_logical
        )

        x_detector_history, x_unsatisfied_fraction = get_detector_history(
            x_syndrome_history,
            x_syndrome_map,
            z_checks,
            n2,
            n_cycles,
            fnc,
            b_initial_state,
        )
        assert HX_decoder.shape[0] == len(x_detector_history)
        x_bpd.decode(x_detector_history)
        bp_converged[i_shot, 1] = x_bpd.converge
        bp_iterations[i_shot, 1] = x_bpd.iter
        unsatisfied_fraction[i_shot, 1] = x_unsatisfied_fraction
        mean_ground[:, 1] += np.asarray(x_pre_readout_history == 0, dtype=float)
        mean_leaked[:, 1] += np.asarray(x_pre_readout_history == 3, dtype=float)

        x_low_weight_error = x_bpd.osdw_decoding
        assert len(x_low_weight_error) == HX.shape[1]
        x_syndrome_history_augmented_guessed = (HX @ x_low_weight_error) % 2
        x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[
            x_logical_row : (x_logical_row + k)
        ]
        if b_initial_state:
            x_state = (x_state + x_initial_state) % 2
        x_state_data_qubits = [x_state[lin_order[q]] for q in data_qubits]
        x_syndrome_final_logical = (lz @ x_state_data_qubits) % 2

        b_x_success = np.array_equal(
            x_syndrome_final_logical_guessed, x_syndrome_final_logical
        )
        if not b_z_success:
            bad_shots_z += 1
        if not b_x_success:
            bad_shots_x += 1
        if not b_z_success or not b_x_success:
            bad_shots += 1
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")
        elif (i_shot % output_step) == (output_step - 1):
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")

    mean_ground /= n_shots
    mean_leaked /= n_shots
    t2 = datetime.now()
    duration = (t2 - t1).total_seconds()
    shot_time = duration / n_shots
    cycle_err = bad_shots / (n_shots * n_cycles)
    shots_err = (cycle_err * (1. - cycle_err) / (n_shots * n_cycles)) ** 0.5

    print(f"\nReadout flip: {b_readout_flip}.")
    print(f"Number of cycles: {n_cycles}.")
    print(
        f"Simulation duration: {round(duration, 2)}s, " f"{round(shot_time, 4)} s/shot."
    )
    print(
        f"Mean Z, X statistics:\n"
        f"BP iterations: {np.mean(bp_iterations, 0)}.\n"
        f"BP converged fraction: {np.mean(bp_converged, 0)}.\n"
        f"Unsatisfied-checks fraction: {np.mean(unsatisfied_fraction, 0)}.\n"
        f"Pre-readout ground state population: {np.mean(mean_ground, 0)}.\n"
        f"Pre-readout leaked state population: {np.mean(mean_leaked, 0)}.\n"
    )

    results = {
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "cycle_err": cycle_err,
        "shots_err": shots_err,
        "bad_shots_z": bad_shots_z,
        "bad_shots_x": bad_shots_x,
        "noise_model": noise_model,
        "mean_ground_z_x": mean_ground,
        "mean_leaked_z_x": mean_leaked,
        "bp_converged_z_x": bp_converged,
        "bp_iterations_z_x": bp_iterations,
        # "unsatisfied_fraction_z_x": unsatisfied_fraction,
    }
    summary = {
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "cycle_err": cycle_err,
        "shots_err": shots_err,
        "mean_ground": np.mean(mean_ground),
        "mean_leaked": np.mean(mean_leaked),
        "bp_converged": np.mean(bp_converged),
        "bp_iterations": np.mean(bp_iterations),
        # "unsatisfied_fraction": np.mean(unsatisfied_fraction),
    }
    return results, summary
