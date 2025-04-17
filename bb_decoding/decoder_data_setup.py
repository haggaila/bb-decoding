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
from bposd.css import css_code
from scipy.sparse import coo_matrix
from scipy.sparse import hstack

from bb_decoding.noise_model import NoiseModel
from bb_decoding.circuit_simulation import (
    generate_linearized_faulty_circuits,
    simulate_x_errors,
    simulate_z_errors,
)


def rank2(A):
    """Takes as input a binary square matrix A.

    Returns:
        The rank of A over the binary field F_2.
    """
    rows, n = A.shape
    X = np.identity(n, dtype=int)

    for i in range(rows):
        y = np.dot(A[i, :], X) % 2
        not_y = (y + 1) % 2
        good = X[:, np.nonzero(not_y)]
        good = good[:, 0, :]
        bad = X[:, np.nonzero(y)]
        bad = bad[:, 0, :]
        if bad.shape[1] > 0:
            bad = np.add(bad, np.roll(bad, 1, axis=1))
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]


def generate_decoder_data(decoder_input: Dict, noise_model: NoiseModel) -> Dict:
    """
    Generates a complete decoder data dictionary for a repeated idle syndrome cycle.

    Args:
        decoder_input: A dictionary defining the decoder data input, with the following keys;
            "code_name": A name defining the code, in the format "n.k.d".
            "n_cycles": Number of idle syndrome cycle repetitions to decode.
            "fnc": Number of final noiseless cycles to add to the full circuit (for validation).
            "unique_id": A uuid to identify the generated decoder data structure.
        noise_model: The noise model to take error probabilities from.

    Returns:
        decoder_data: The dictionary data structure of the code and decoding.
    """

    # Parameters of a Bivariate Bicycle (BB) code
    # see Section 4 of https://arxiv.org/pdf/2308.07915.pdf for notations
    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^ell = 1
    # y^m = 1
    # A = x^{a_1} + y^{a_2} + y^{a_3}
    # B = y^{b_1} + x^{b_2} + x^{b_3}

    s_code_name: str = decoder_input["code_name"]
    n_cycles: int = decoder_input["n_cycles"]
    fnc: int = decoder_input["fnc"]

    if s_code_name == "72.12.6":
        ell, m = 6, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
        d = 6
    elif s_code_name == "144.12.12":
        ell, m = 12, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
        d = 12
    elif s_code_name == "288.12.18":
        ell, m = 12, 12
        a1, a2, a3 = 3, 2, 7
        b1, b2, b3 = 3, 1, 2
        d = 18
    elif s_code_name == "784.24.24":
        ell, m = 28, 14
        a1, a2, a3 = 26, 6, 8
        b1, b2, b3 = 7, 9, 20
        d = 24
    else:
        raise Exception("Unsupported code.")
    # code length
    n = 2 * m * ell
    n2 = m * ell

    # Compute check matrices of X- and Z-checks

    # cyclic shift matrices
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    I = np.identity(ell * m, dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for i in range(m):
        y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))

    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2

    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    # number of logical qubits
    k = n - rank2(hx) - rank2(hz)

    qcode = css_code(hx, hz, d)
    print("Testing CSS code...")
    qcode.test()
    print("Done")

    lz = qcode.lz
    lx = qcode.lx

    # Give a name to each qubit
    # Define a linear order on the set of qubits
    lin_order = {}
    data_qubits = []
    x_checks = []
    z_checks = []
    data_qubit_indices = []
    cnt = 0
    for i in range(n2):
        node_name = ("XC", i)
        x_checks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1
    for i in range(n2):
        node_name = ("DL", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        data_qubit_indices.append(cnt)
        cnt += 1
    for i in range(n2):
        node_name = ("DR", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        data_qubit_indices.append(cnt)
        cnt += 1
    for i in range(n2):
        node_name = ("ZC", i)
        z_checks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1

    # compute the list of neighbors of each check qubit in the Tanner graph
    nbs = {}
    # iterate over X checks
    for i in range(n2):
        check_name = ("XC", i)
        # left data qubits
        nbs[(check_name, 0)] = ("DL", np.nonzero(A1[i, :])[0][0])
        nbs[(check_name, 1)] = ("DL", np.nonzero(A2[i, :])[0][0])
        nbs[(check_name, 2)] = ("DL", np.nonzero(A3[i, :])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("DR", np.nonzero(B1[i, :])[0][0])
        nbs[(check_name, 4)] = ("DR", np.nonzero(B2[i, :])[0][0])
        nbs[(check_name, 5)] = ("DR", np.nonzero(B3[i, :])[0][0])

    # iterate over Z checks
    for i in range(n2):
        check_name = ("ZC", i)
        # left data qubits
        nbs[(check_name, 0)] = ("DL", np.nonzero(B1[:, i])[0][0])
        nbs[(check_name, 1)] = ("DL", np.nonzero(B2[:, i])[0][0])
        nbs[(check_name, 2)] = ("DL", np.nonzero(B3[:, i])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("DR", np.nonzero(A1[:, i])[0][0])
        nbs[(check_name, 4)] = ("DR", np.nonzero(A2[:, i])[0][0])
        nbs[(check_name, 5)] = ("DR", np.nonzero(A3[:, i])[0][0])

    # syndrome cycle with 7 CNOT rounds
    # sX and sZ define the order in which X-check and Z-check qubit
    # is coupled with the neighboring data qubits
    # We label the six neighbors of each check qubit in the Tanner graph
    # by integers 0,1,...,5
    x_check_coupling = [-1, 1, 4, 3, 5, 0, 2]
    z_check_coupling = [3, 5, 0, 1, 2, 4, -1]

    cycle = []
    U = np.identity(2 * n, dtype=int)
    # round 0: prep xchecks, CNOT zchecks and data
    t = 0
    for q in x_checks:
        cycle.append(("PX", q))
    data_qubits_cnoted_in_this_round = []
    assert not (z_check_coupling[t] == -1)
    for target in z_checks:
        direction = z_check_coupling[t]
        control = nbs[(target, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        data_qubits_cnoted_in_this_round.append(control)
        cycle.append(("CX", control, target))
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("ID_S", q))  # Short idle gate (parallel to CNOT)

    # round 1-5: CNOT xchecks and data, CNOT zchecks and data
    for t in range(1, 6):
        assert not (x_check_coupling[t] == -1)
        for control in x_checks:
            direction = x_check_coupling[t]
            target = nbs[(control, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CX", control, target))
        assert not (z_check_coupling[t] == -1)
        for target in z_checks:
            direction = z_check_coupling[t]
            control = nbs[(target, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CX", control, target))

    # round 6: CNOT xchecks and data, measure Z checks
    t = 6
    for q in z_checks:
        cycle.append(("MZ", q))
    assert not (x_check_coupling[t] == -1)
    data_qubits_cnoted_in_this_round = []
    for control in x_checks:
        direction = x_check_coupling[t]
        target = nbs[(control, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        cycle.append(("CX", control, target))
        data_qubits_cnoted_in_this_round.append(target)
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("ID_L", q))  # Long idle gate (parallel to readout)

    # round 7: all data qubits are idle, Prep Z checks, Meas X checks
    for q in data_qubits:
        cycle.append(("ID_L", q))  # Long idle gate (parallel to readout)
    for q in x_checks:
        cycle.append(("MX", q))
    for q in z_checks:
        cycle.append(("PZ", q))

    # full syndrome measurement circuit
    cycle_repeated = n_cycles * cycle

    # test the syndrome measurement circuit

    # implement syndrome measurements using the sequential depth-12 circuit
    V = np.identity(2 * n, dtype=int)
    # first measure all X checks
    for t in range(7):
        if not (x_check_coupling[t] == -1):
            for control in x_checks:
                direction = x_check_coupling[t]
                target = nbs[(control, direction)]
                V[lin_order[target], :] = (
                    V[lin_order[target], :] + V[lin_order[control], :]
                ) % 2
    # next measure all Z checks
    for t in range(7):
        if not (z_check_coupling[t] == -1):
            for target in z_checks:
                direction = z_check_coupling[t]
                control = nbs[(target, direction)]
                V[lin_order[target], :] = (
                    V[lin_order[target], :] + V[lin_order[control], :]
                ) % 2

    if np.array_equal(U, V):
        print("Circuit test: OK.")
    else:
        print("Circuit test: FAIL!")
        exit()

    decoder_data = decoder_input.copy()
    decoder_data.update(
        {
            "n": n,
            "k": k,
            "lin_order": lin_order,
            "data_qubits": data_qubits,
            "x_checks": x_checks,
            "z_checks": z_checks,
            "nbs": nbs,
            "n_neighbors": 6,
        }
    )

    # Compute decoding matrices
    (
        z_circuits,
        z_fault_probs,
        x_circuits,
        x_fault_probs,
    ) = generate_linearized_faulty_circuits(cycle_repeated, noise_model)

    # execute each noisy circuit and compute the syndrome
    # we add two noiseless syndrome cycles at the end
    print("Computing syndrome histories for single-Z-type-fault circuits...")
    cnt = 0
    HZdict = {}
    for circ in z_circuits:
        full_circuit = circ.copy()
        for i in range(fnc):
            full_circuit += cycle
        syndrome_history, state, syndrome_map = simulate_z_errors(
            full_circuit, n2, lin_order
        )
        assert len(syndrome_history) == n2 * (n_cycles + fnc)
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lx @ state_data_qubits) % 2
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
        for c in x_checks:
            pos = syndrome_map[c]
            assert len(pos) == (n_cycles + fnc)
            for row in range(1, n_cycles + fnc):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= 2
        syndrome_history_augmented = np.hstack(
            [syndrome_history, syndrome_final_logical]
        )
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HZdict:
            HZdict[supp].append(cnt)
        else:
            HZdict[supp] = [cnt]
        cnt += 1
    z_logical_row = n2 * (n_cycles + fnc)
    print("Done.")

    # if a subset of columns of HZ are equal, retain only one of these columns
    print("Computing effective noise model for the Z-type-faults decoder...")
    print("Number of distinct Z-syndrome histories: ", len(HZdict))
    HZ = []
    HZ_decoder = []
    z_probs = []
    for supp in HZdict:
        new_column = np.zeros((n2 * (n_cycles + fnc) + k, 1), dtype=int)
        new_column_short = np.zeros((n2 * (n_cycles + fnc), 1), dtype=int)
        new_column[list(supp), 0] = 1
        new_column_short[:, 0] = new_column[0:z_logical_row, 0]
        HZ.append(coo_matrix(new_column))
        HZ_decoder.append(coo_matrix(new_column_short))
        z_probs.append(np.sum([z_fault_probs[i] for i in HZdict[supp]]))
    print("Done.")
    HZ = hstack(HZ)
    HZ_decoder = hstack(HZ_decoder)
    print("Decoding matrix HZ sparseness:")
    print("max col weight=", np.max(np.sum(HZ_decoder, 0)))
    print("max row weight=", np.max(np.sum(HZ_decoder, 1)))

    # execute each noisy circuit and compute the syndrome
    # we add two noiseless syndrome cycles at the end
    print("Computing syndrome histories for single-X-type-fault circuits...")
    cnt = 0
    HXdict = {}
    for circ in x_circuits:
        full_circuit = circ.copy()
        for i in range(fnc):
            full_circuit += cycle
        syndrome_history, state, syndrome_map = simulate_x_errors(
            full_circuit, n2, lin_order
        )
        assert len(syndrome_history) == n2 * (n_cycles + fnc)
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lz @ state_data_qubits) % 2
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
        for c in z_checks:
            pos = syndrome_map[c]
            assert len(pos) == (n_cycles + fnc)
            for row in range(1, n_cycles + fnc):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= 2
        syndrome_history_augmented = np.hstack(
            [syndrome_history, syndrome_final_logical]
        )
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HXdict:
            HXdict[supp].append(cnt)
        else:
            HXdict[supp] = [cnt]
        cnt += 1
    x_logical_row = n2 * (n_cycles + fnc)
    print("Done.")

    # if a subset of columns of H are equal, retain only one of these columns
    print("Computing effective noise model for the X-type-faults decoder...")
    print("Number of distinct X-syndrome histories: ", len(HXdict))
    HX = []
    HX_decoder = []
    x_probs = []
    for supp in HXdict:
        new_column = np.zeros((n2 * (n_cycles + fnc) + k, 1), dtype=int)
        new_column_short = np.zeros((n2 * (n_cycles + fnc), 1), dtype=int)
        new_column[list(supp), 0] = 1
        new_column_short[:, 0] = new_column[0:x_logical_row, 0]
        HX.append(coo_matrix(new_column))
        HX_decoder.append(coo_matrix(new_column_short))
        x_probs.append(np.sum([x_fault_probs[i] for i in HXdict[supp]]))
    print("Done.")
    HX = hstack(HX)
    HX_decoder = hstack(HX_decoder)
    print("Decoding matrix HX sparseness:")
    print("max col weight: ", np.max(np.sum(HX_decoder, 0)))
    print("max row weight: ", np.max(np.sum(HX_decoder, 1)))

    # save decoding matrices
    decoder_data["data_qubit_indices"] = data_qubit_indices
    decoder_data["HX_decoder"] = HX_decoder
    decoder_data["HZ_decoder"] = HZ_decoder
    decoder_data["x_probs"] = x_probs
    decoder_data["z_probs"] = z_probs
    decoder_data["cycle"] = cycle
    decoder_data["HX"] = HX
    decoder_data["HZ"] = HZ
    decoder_data["lx"] = lx
    decoder_data["lz"] = lz
    decoder_data["z_logical_row"] = z_logical_row
    decoder_data["x_logical_row"] = x_logical_row
    decoder_data["ell"] = ell
    decoder_data["m"] = m
    decoder_data["a1"] = a1
    decoder_data["a2"] = a2
    decoder_data["a3"] = a3
    decoder_data["b1"] = b1
    decoder_data["b2"] = b2
    decoder_data["b3"] = b3
    decoder_data["noise_model"] = noise_model
    decoder_data["x_check_coupling"] = x_check_coupling
    decoder_data["z_check_coupling"] = z_check_coupling
    print("Done.")
    return decoder_data
