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


def _apply_readout(
    gate,
    check_name,
    lin_order,
    n2,
    state,
    syndrome_history,
    syndrome_map,
    syn_cnt,
    noise_model=None,
    b_readout_flip=False,
    b_errors_on=False,
    err_rngs=None,
    pre_readout_history=None,
    prep_correlated_errors=None,
    leaked_population=0.,
):
    assert len(gate) == 2 and gate[1][0] == check_name
    if noise_model is not None:
        b_is_leaky = noise_model.is_leaky
        b_state_sim = noise_model.is_state_dependent
    else:
        b_is_leaky = b_state_sim = False
    q = lin_order[gate[1]]
    q_state = state[q]
    if b_state_sim:
        if b_errors_on:
            # Simulating the nonideal syndrome cycles
            b_incoming_bit = False
            # b_incoming_bit indicates for the flip protocol whether a flip is needed based on
            # previous readout.
            if b_readout_flip and len(syndrome_history) >= n2:
                # If there is a history for the check qubits from the previous readout,
                # assign b_incoming_bit according to whether the check was reading 1.
                b_incoming_bit = syndrome_history[-n2]
                if b_incoming_bit and q_state <= 1:
                    # If the check was previously reading 1, and it is currently not leaked, flip.
                    err_rate = noise_model.flip
                    if not (err_rngs[0] <= err_rate):
                        # Note that an error in this case means NO flip!
                        q_state = (q_state + 1) % 2
            if pre_readout_history is not None:
                pre_readout_history.append(q_state)
                # This records the actual state just before readout, only in non-ideal cycles.
            if b_is_leaky:
                if q_state == 1:
                    # A check qubit only leaks in this model if it is in the state |1>.
                    err_rate = noise_model.leak
                    if err_rngs[2] <= err_rate:
                        q_state = 3  # This will be measured as 1
                elif q_state == 3:
                    # A leaked check qubit can seep back to the state |1>.
                    err_rate = noise_model.seep
                    if err_rngs[2] <= err_rate:
                        q_state = 1
            meas = noise_model.meas
            bias = noise_model.bias
            err_rate = meas + bias if q_state else meas - bias
            if err_rngs[1] <= err_rate and q_state <= 1:
                # A leaked qubit must remain with the state value 3, so it is excluded.
                q_state = (q_state + 1) % 2
                # Note that a seeped qubit (put back to the state 1), can still have in this model
                # a readout error and measure as 0. This is probably not really significant.
                err_rate = noise_model.corr
                if err_rngs[3] <= err_rate:
                    prep_correlated_errors[q] = 1
            state[q] = q_state
            if q_state == 3:
                q_state = 1
                # This ensures that leaked qubits (with q_state == 3) will be measured as 1.
            if b_readout_flip and b_incoming_bit:
                q_state = (q_state + 1) % 2
        elif leaked_population and q_state == 1:
            # Check qubits in the state 1 could be promoted to a leaked state for the remaining
            # simulation, to have an initial leaked population close to its steady state value.
            # Note that leaked_population controls the probability and allows to turn
            # this behavior off completely for the final noiseless cycles.
            # The factor of two below is because leaked_population is conveniently expressed from
            # the overall check population, but we here set leaked qubits only from those in the
            # state 1 for compatibility with the readout result.
            err_rate = 2 * leaked_population
            if err_rngs[2] <= err_rate:
                state[q] = 3

    syndrome_history.append(q_state)
    if gate[1] in syndrome_map:
        syndrome_map[gate[1]].append(syn_cnt)
    else:
        syndrome_map[gate[1]] = [syn_cnt]


def simulate_errors(
    full_circuit,
    n2,
    lin_order,
    rng: np.random.Generator,
    z_initial_state=None,
    x_initial_state=None,
    noise_model=None,
    b_readout_flip=False,
    leaked_population=.0,
):
    z_prep_correlated_errors = {}
    z_pre_readout_history = []
    z_syndrome_history = []
    z_syndrome_map = {}
    # keys: x_checks, vals: positions in the syndrome history array

    x_prep_correlated_errors = {}
    x_pre_readout_history = []
    x_syndrome_history = []
    x_syndrome_map = {}
    # keys: z_checks, vals: positions in the syndrome history array

    if z_initial_state is None:
        z_state = np.zeros(2 * 2 * n2, dtype=int)
    else:
        z_state = z_initial_state.copy()
    if x_initial_state is None:
        x_state = np.zeros(2 * 2 * n2, dtype=int)
    else:
        x_state = x_initial_state.copy()

    n_gates = len(full_circuit)
    err_randoms = rng.uniform(size=(n_gates, 4))
    # We pre-allocate the array of random uniform values that will be used to determine
    # the occurrence of an error in each gate (currently CNOTs and measurements).
    # We need up to 4 per readout gate; for the biased measurement, leakage or seepage, flip,
    # and correlated prep error (that is determined at readout).
    # We need up to 2 per CNOT gate; if the check is leaked, independent possible X and Z errors
    # as backaction on the data qubit.

    b_errors_on = False
    z_syn_cnt = 0
    x_syn_cnt = 0
    for i_gate, gate in enumerate(full_circuit):
        s_gate_name = gate[0]
        err_rngs = err_randoms[i_gate, :]
        if s_gate_name == "MX":
            _apply_readout(
                gate,
                "XC",
                lin_order,
                n2,
                z_state,
                z_syndrome_history,
                z_syndrome_map,
                z_syn_cnt,
                noise_model,
                b_readout_flip,
                b_errors_on,
                err_rngs,
                z_pre_readout_history,
                z_prep_correlated_errors,
                leaked_population,
            )
            z_syn_cnt += 1
        elif s_gate_name == "MZ":
            _apply_readout(
                gate,
                "ZC",
                lin_order,
                n2,
                x_state,
                x_syndrome_history,
                x_syndrome_map,
                x_syn_cnt,
                noise_model,
                b_readout_flip,
                b_errors_on,
                err_rngs,
                x_pre_readout_history,
                x_prep_correlated_errors,
                leaked_population,
            )
            x_syn_cnt += 1
        elif s_gate_name == "ON":
            b_errors_on = True
            # Allows to turn on errors following the initial noiseless syndrome cycle.
        elif s_gate_name == "OFF":
            b_errors_on = False
            # Allows to turn off errors for the final noiseless syndrome cycles.
            if noise_model.is_leaky:
                # Guarantees that all leaked checks are put back to the state 1, and will be
                # initialized to 0 in the preparation steps.
                x_state = x_state % 2
                z_state = z_state % 2
            leaked_population = .0  # Guarantees no more qubits are initialized as leaked
        else:
            apply_gate_in_z_simulation(
                gate, z_state, lin_order, noise_model, x_state, err_rngs, z_prep_correlated_errors
            )
            apply_gate_in_x_simulation(
                gate, x_state, lin_order, noise_model, z_state, err_rngs, x_prep_correlated_errors
            )

    return (
        np.array(z_pre_readout_history, dtype=int),
        np.array(z_syndrome_history, dtype=int),
        z_syndrome_map,
        z_state,
        np.array(x_pre_readout_history, dtype=int),
        np.array(x_syndrome_history, dtype=int),
        x_syndrome_map,
        x_state,
    )


def simulate_z_errors(full_circuit, n2, lin_order, initial_state=None):
    # we only look at the action of the circuit on Z errors; 0 means no error, 1 means error
    syndrome_history = []
    # keys: x_checks, vals: positions in the syndrome history array
    syndrome_map = {}
    if initial_state is None:
        state = np.zeros(2 * 2 * n2, dtype=int)
    else:
        state = initial_state.copy()
    syn_cnt = 0
    for gate in full_circuit:
        if gate[0] == "MX":
            _apply_readout(
                gate,
                "XC",
                lin_order,
                n2,
                state,
                syndrome_history,
                syndrome_map,
                syn_cnt,
            )
            syn_cnt += 1
        else:
            apply_gate_in_z_simulation(gate, state, lin_order)

    return np.array(syndrome_history, dtype=int), state, syndrome_map


def apply_gate_in_z_simulation(
    gate, state, lin_order, noise_model=None, other_state=None, err_rngs=None,
    prep_correlated_errors=None
):
    s_gate_name = gate[0]
    if s_gate_name == "CX":
        assert len(gate) == 3
        control = lin_order[gate[1]]
        target = lin_order[gate[2]]
        control_state = state[control]
        target_state = state[target]
        if control_state <= 1:
            # Check qubit is not leaked, apply gate
            state[control] = (target_state + control_state) % 2
        else:
            # Check qubit is leaked; Skip state change, apply possible backaction on data
            err_rate = noise_model.back
            if err_rngs[0] <= err_rate:
                state[target] = (state[target] + 1) % 2
            if err_rngs[1] <= err_rate:
                other_state[target] = (other_state[target] + 1) % 2
    elif s_gate_name == "PX":
        assert len(gate) == 2
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            # Check qubit is not leaked, apply init
            state[q] = 0
            if prep_correlated_errors is not None:
                prep_err = prep_correlated_errors.pop(q, 0)
                if prep_err:
                    state[q] = 1
    elif s_gate_name in ["Z", "Y"]:
        assert len(gate) == 2
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["ZX", "YX"]:
        assert len(gate) == 3
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["XZ", "XY"]:
        assert len(gate) == 3
        q = lin_order[gate[2]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["ZZ", "YY", "YZ", "ZY"]:
        assert len(gate) == 3
        q1 = lin_order[gate[1]]
        q1_state = state[q1]
        if q1_state <= 1:
            state[q1] = (q1_state + 1) % 2
        q2 = lin_order[gate[2]]
        q2_state = state[q2]
        if q2_state <= 1:
            state[q2] = (q2_state + 1) % 2


def simulate_x_errors(full_circuit, n2, lin_order, initial_state=None):
    # we only look at the action of the circuit on X errors; 0 means no error, 1 means error
    syndrome_history = []
    # keys: z_checks, vals: positions in the syndrome history array
    syndrome_map = {}
    if initial_state is None:
        state = np.zeros(2 * 2 * n2, dtype=int)
    else:
        state = initial_state.copy()
    syn_cnt = 0
    for gate in full_circuit:
        if gate[0] == "MZ":
            _apply_readout(
                gate,
                "ZC",
                lin_order,
                n2,
                state,
                syndrome_history,
                syndrome_map,
                syn_cnt,
            )
            syn_cnt += 1
        else:
            apply_gate_in_x_simulation(gate, state, lin_order)

    return np.array(syndrome_history, dtype=int), state, syndrome_map


def apply_gate_in_x_simulation(
    gate, state, lin_order, noise_model=None, other_state=None, err_rngs=None,
    prep_correlated_errors=None,
):
    s_gate_name = gate[0]
    if s_gate_name == "CX":
        assert len(gate) == 3
        control = lin_order[gate[1]]
        target = lin_order[gate[2]]
        control_state = state[control]
        target_state = state[target]
        if target_state <= 1:
            # Check qubit is not leaked, apply gate
            state[target] = (target_state + control_state) % 2
        else:
            # Check qubit is leaked; Skip state change, apply possible backaction on data
            err_rate = noise_model.back
            if err_rngs[0] <= err_rate:
                state[control] = (state[control] + 1) % 2
            if err_rngs[1] <= err_rate:
                other_state[control] = (other_state[control] + 1) % 2
    elif s_gate_name == "PZ":
        assert len(gate) == 2
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            # Check qubit is not leaked, apply init
            state[q] = 0
            if prep_correlated_errors is not None:
                prep_err = prep_correlated_errors.pop(q, 0)
                if prep_err:
                    state[q] = 1
    elif s_gate_name in ["X", "Y"]:
        assert len(gate) == 2
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["XZ", "YZ"]:
        assert len(gate) == 3
        q = lin_order[gate[1]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["ZX", "ZY"]:
        assert len(gate) == 3
        q = lin_order[gate[2]]
        q_state = state[q]
        if q_state <= 1:
            state[q] = (q_state + 1) % 2
    elif s_gate_name in ["XX", "YY", "XY", "YX"]:
        assert len(gate) == 3
        q1 = lin_order[gate[1]]
        q1_state = state[q1]
        if q1_state <= 1:
            state[q1] = (q1_state + 1) % 2
        q2 = lin_order[gate[2]]
        q2_state = state[q2]
        if q2_state <= 1:
            state[q2] = (q2_state + 1) % 2


def get_detector_history(
    syndrome_history, syndrome_map, checks, n2, n_cycles, fnc, b_initial_state
):
    inc = 1 if b_initial_state else 0
    assert len(syndrome_history) == n2 * (n_cycles + fnc + inc)
    # apply syndrome sparsification map
    detector_history = syndrome_history.copy()
    for c in checks:
        pos = syndrome_map[c]
        assert len(pos) == (n_cycles + fnc + inc)
        for row in range(1, n_cycles + fnc + inc):
            detector_history[pos[row]] += syndrome_history[pos[row - 1]]
    detector_history %= 2
    if b_initial_state:
        detector_history = detector_history[n2:]
    unsatisfied_fraction = np.count_nonzero(syndrome_history[inc * n2 : -fnc * n2]) / (
        n_cycles * n2
    )
    return detector_history, unsatisfied_fraction


def generate_noisy_circuit(cycle_repeated, noise_model, rng: np.random.Generator):
    circ = []
    n_gates = len(cycle_repeated)
    err_randoms = rng.uniform(size=n_gates)
    # We pre-allocate the array of random uniform values that will be used to determine
    # the occurrence of an error in each gate

    for gate, err_rng in zip(cycle_repeated, err_randoms):
        s_gate_name = gate[0]
        assert s_gate_name in ["CX", "ID_S", "ID_L", "PX", "PZ", "MX", "MZ"]
        if s_gate_name == "CX":
            circ.append(gate)
            if err_rng <= noise_model.cnot:
                error_type = rng.integers(15)
                if error_type == 0:
                    circ.append(("X", gate[1]))
                elif error_type == 1:
                    circ.append(("Y", gate[1]))
                elif error_type == 2:
                    circ.append(("Z", gate[1]))
                elif error_type == 3:
                    circ.append(("X", gate[2]))
                elif error_type == 4:
                    circ.append(("Y", gate[2]))
                elif error_type == 5:
                    circ.append(("Z", gate[2]))
                elif error_type == 6:
                    circ.append(("XX", gate[1], gate[2]))
                elif error_type == 7:
                    circ.append(("YY", gate[1], gate[2]))
                elif error_type == 8:
                    circ.append(("ZZ", gate[1], gate[2]))
                elif error_type == 9:
                    circ.append(("XY", gate[1], gate[2]))
                elif error_type == 10:
                    circ.append(("YX", gate[1], gate[2]))
                elif error_type == 11:
                    circ.append(("YZ", gate[1], gate[2]))
                elif error_type == 12:
                    circ.append(("ZY", gate[1], gate[2]))
                elif error_type == 13:
                    circ.append(("XZ", gate[1], gate[2]))
                elif error_type == 14:
                    circ.append(("ZX", gate[1], gate[2]))
        elif s_gate_name[:2] == "ID":
            err_rate = noise_model.id_s if s_gate_name[-1] == "S" else noise_model.id_l
            if err_rng <= err_rate:
                ptype = rng.integers(3)
                if ptype == 0:
                    circ.append(("X", gate[1]))
                if ptype == 1:
                    circ.append(("Y", gate[1]))
                if ptype == 2:
                    circ.append(("Z", gate[1]))
        elif s_gate_name == "PX":
            circ.append(gate)
            if err_rng <= noise_model.prep:
                circ.append(("Z", gate[1]))
        elif s_gate_name == "PZ":
            circ.append(gate)
            if err_rng <= noise_model.prep:
                circ.append(("X", gate[1]))
        elif s_gate_name == "MX":
            if not noise_model.is_state_dependent:
                if err_rng <= noise_model.meas:
                    circ.append(("Z", gate[1]))
            circ.append(gate)
        elif s_gate_name == "MZ":
            if not noise_model.is_state_dependent:
                if err_rng <= noise_model.meas:
                    circ.append(("X", gate[1]))
            circ.append(gate)
    return circ


def generate_linearized_faulty_circuits(cycle_repeated, noise_model):
    print("Generating noisy circuits with a single Z-type faulty gate...")
    z_fault_probs = []
    z_circuits = []
    head = []
    tail = cycle_repeated.copy()
    for gate in cycle_repeated:
        s_gate_name = gate[0]
        assert s_gate_name in ["CX", "ID_S", "ID_L", "PX", "PZ", "MX", "MZ"]
        if s_gate_name == "MX":
            assert len(gate) == 2
            z_circuits.append(head + [("Z", gate[1])] + tail)
            z_fault_probs.append(noise_model.meas)
        # move the gate from tail to head
        head.append(gate)
        tail.pop(0)
        assert cycle_repeated == (head + tail)
        if s_gate_name == "CX":
            assert len(gate) == 3
            # add error on the control qubit
            z_circuits.append(head + [("Z", gate[1])] + tail)
            z_fault_probs.append(noise_model.cnot * 4 / 15)
            # add error on the target qubit
            z_circuits.append(head + [("Z", gate[2])] + tail)
            z_fault_probs.append(noise_model.cnot * 4 / 15)
            # add ZZ error on the control and the target qubits
            z_circuits.append(head + [("ZZ", gate[1], gate[2])] + tail)
            z_fault_probs.append(noise_model.cnot * 4 / 15)
        elif s_gate_name[:2] == "ID":
            assert len(gate) == 2
            z_circuits.append(head + [("Z", gate[1])] + tail)
            err_rate = noise_model.id_s if s_gate_name[-1] == "S" else noise_model.id_l
            z_fault_probs.append(err_rate * 2 / 3)
        elif s_gate_name == "PX":
            assert len(gate) == 2
            z_circuits.append(head + [("Z", gate[1])] + tail)
            z_fault_probs.append(noise_model.prep)
    print(f"Done. Number of noisy Z circuits: {len(z_circuits)}.")

    print("Generating noisy circuits with a single X-type faulty gate...")
    x_fault_probs = []
    x_circuits = []
    head = []
    tail = cycle_repeated.copy()
    for gate in cycle_repeated:
        s_gate_name = gate[0]
        assert s_gate_name in ["CX", "ID_S", "ID_L", "PX", "PZ", "MX", "MZ"]
        if s_gate_name == "MZ":
            assert len(gate) == 2
            x_circuits.append(head + [("X", gate[1])] + tail)
            x_fault_probs.append(noise_model.meas)
        # move the gate from tail to head
        head.append(gate)
        tail.pop(0)
        assert cycle_repeated == (head + tail)
        if s_gate_name == "CX":
            assert len(gate) == 3
            # add error on the control qubit
            x_circuits.append(head + [("X", gate[1])] + tail)
            x_fault_probs.append(noise_model.cnot * 4 / 15)
            # add error on the target qubit
            x_circuits.append(head + [("X", gate[2])] + tail)
            x_fault_probs.append(noise_model.cnot * 4 / 15)
            # add XX error on the control and the target qubits
            x_circuits.append(head + [("XX", gate[1], gate[2])] + tail)
            x_fault_probs.append(noise_model.cnot * 4 / 15)
        elif s_gate_name[:2] == "ID":
            assert len(gate) == 2
            x_circuits.append(head + [("X", gate[1])] + tail)
            err_rate = noise_model.id_s if s_gate_name[-1] == "S" else noise_model.id_l
            x_fault_probs.append(err_rate * 2 / 3)
        elif s_gate_name == "PZ":
            assert len(gate) == 2
            x_circuits.append(head + [("X", gate[1])] + tail)
            x_fault_probs.append(noise_model.prep)
    print(f"Done. Number of noisy X circuits: {len(x_circuits)}.")

    return z_circuits, z_fault_probs, x_circuits, x_fault_probs
