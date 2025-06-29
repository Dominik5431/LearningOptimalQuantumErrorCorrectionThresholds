import numpy as np
import stim
import matplotlib.pyplot as plt
import pymatching
from tqdm import tqdm

import stim
import pymatching


def surfacecode_coords(circuit, distance, logical_qubit_number, initilization_type):
    i = (logical_qubit_number - 1) * (2 * distance + 1) ** 2
    data_qubits = []
    ancialla_qubits_X = []
    ancialla_qubits_Z = []
    for row in range(2 * distance + 1):
        for col in range(2 * distance + 1):
            if row % 2 == 1 and col % 2 == 1:
                circuit.append_operation("QUBIT_COORDS", i,
                                         (col + (logical_qubit_number - 1) * (2 * distance + 1), row))
                data_qubits.append(i)
            elif row % 2 == 0 and col % 2 == 0 and (col + row) % 4 == 0 and col != 0 and col != 2 * distance:
                circuit.append_operation("QUBIT_COORDS", i,
                                         (col + (logical_qubit_number - 1) * (2 * distance + 1), row))
                ancialla_qubits_X.append(i)
            elif row % 2 == 0 and col % 2 == 0 and (col + row) % 4 != 0 and row != 0 and row != 2 * distance:
                circuit.append_operation("QUBIT_COORDS", i,
                                         (col + (logical_qubit_number - 1) * (2 * distance + 1), row))
                ancialla_qubits_Z.append(i)
            i += 1
    return circuit, data_qubits, ancialla_qubits_X, ancialla_qubits_Z


def init(circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z, initilization_type):
    if initilization_type == "Z":
        circuit.append_operation("R", data_qubits)
    elif initilization_type == "X":
        circuit.append_operation("RX", data_qubits)
    elif initilization_type == "Y":
        circuit.append_operation("RY", data_qubits)
    circuit.append_operation("R", ancilla_qubits_X)
    circuit.append_operation("R", ancilla_qubits_Z)
    circuit.append_operation("TICK")


def S_X(circuit: stim.Circuit, data_qubits, ancilla_qubits_X, distance, before_measure_flip_probability,
        after_reset_flip_probability, after_clifford_polarisation, initialization_type):
    if initialization_type == "Z":
        pattern = [-(2 * distance + 2), 2 * distance, -2 * distance, 2 * distance + 2]
    elif initialization_type == "X" or initialization_type == "Y":
        pattern = [-(2 * distance + 2), (2 * distance), -2 * distance, 2 * distance + 2]
    pattern = [2 * distance + 2, 2 * distance, -2 * distance, -2 * distance - 2]
    for i in ancilla_qubits_X:
        circuit.append_operation("X_ERROR", i, after_reset_flip_probability)
        circuit.append_operation("H", i)
        circuit.append_operation("DEPOLARIZE1", i, after_clifford_polarisation)
        circuit.append_operation("TICK")
        for j in range(4):
            if i + pattern[j] in data_qubits:
                circuit.append_operation("CNOT", [i, i + pattern[j]])
                circuit.append_operation("DEPOLARIZE2", [i, i + pattern[j]], after_clifford_polarisation)
        circuit.append_operation("TICK")
        circuit.append_operation("H", i)
        circuit.append_operation("DEPOLARIZE1", i, after_clifford_polarisation)
        circuit.append_operation("X_ERROR", i, before_measure_flip_probability)
        circuit.append_operation("MR", i)


def S_Z(circuit: stim.Circuit, data_qubits, ancilla_qubits_Z, distance, before_measure_flip_probability,
        after_reset_flip_probability, after_clifford_polarisation, initialization_type):
    if initialization_type == "Z" or initialization_type == "Y":
        pattern = [-(2 * distance + 2), 2 * distance, -2 * distance, 2 * distance + 2]
    elif initialization_type == "X":
        pattern = [-(2 * distance + 2), (2 * distance), -2 * distance, 2 * distance + 2]
    pattern = [2 * distance + 2, -2 * distance, 2 * distance, -2 * distance - 2]
    for i in ancilla_qubits_Z:
        circuit.append_operation("X_ERROR", i, after_reset_flip_probability)
        circuit.append_operation("I", i)
        circuit.append_operation("DEPOLARIZE1", i, after_clifford_polarisation)
        for j in range(4):
            if i + pattern[j] in data_qubits:
                circuit.append_operation("CNOT", [i + pattern[j], i])
                circuit.append_operation("DEPOLARIZE2", [i + pattern[j], i], after_clifford_polarisation)
        circuit.append_operation("TICK")
        circuit.append_operation("I", i)
        circuit.append_operation("DEPOLARIZE1", i, after_clifford_polarisation)
        circuit.append_operation("X_ERROR", i, before_measure_flip_probability)
        circuit.append_operation("MR", i)


def surfacecode_init(circuit, distance, logical_qubit_number, initilization_type):
    circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z = surfacecode_coords(circuit, distance,
                                                                                  logical_qubit_number,
                                                                                  initilization_type)
    init(circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z, initilization_type)
    S_X(circuit, data_qubits, ancilla_qubits_X, distance, 0, 0, 0, initilization_type)
    S_Z(circuit, data_qubits, ancilla_qubits_Z, distance, 0, 0, 0, initilization_type)
    return circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z


def data_qubit_noise(circuit: stim.Circuit, data_qubits, before_round_data_polarisation):
    circuit.append_operation("DEPOLARIZE1", data_qubits, before_round_data_polarisation)
    # circuit.append_operation("X_ERROR", data_qubits, before_round_data_polarisation/1E20)
    # circuit.append_operation("Z_ERROR", data_qubits, before_round_data_polarisation/1E20)
    # circuit.append_operation("Y_ERROR", data_qubits, before_round_data_polarisation)
    # circuit.append_operation("X_ERROR", data_qubits, before_round_data_polarisation)
    # circuit.append_operation("Z_ERROR", data_qubits, before_round_data_polarisation)
    circuit.append("TICK")


def stabilizer_measurement(circuit, data_qubits, ancilla_qubits_Z, ancilla_qubits_X, distance,
                           before_measure_flip_probability, after_reset_flip_probability, after_clifford_polarisation,
                           initilization_type, logical_qubit_number, round, number_total_qubits=1):
    S_X(circuit, data_qubits, ancilla_qubits_X, distance, before_measure_flip_probability, after_reset_flip_probability,
        after_clifford_polarisation, initilization_type)
    S_Z(circuit, data_qubits, ancilla_qubits_Z, distance, before_measure_flip_probability, after_reset_flip_probability,
        after_clifford_polarisation, initilization_type)
    ancillas = ancilla_qubits_X + ancilla_qubits_Z
    for i in range(1, len(ancillas) + 1):
        circuit.append_operation("DETECTOR", [stim.target_rec(-len(ancillas) + i - 1), stim.target_rec(
            -(2 + (number_total_qubits - 1)) * len(ancillas) + i - 1)], (
                                     int((ancillas[i - 1] - (logical_qubit_number - 1) * (2 * distance + 1) ** 2) % (
                                             2 * distance + 1) + (logical_qubit_number - 1) * (2 * distance + 1)),
                                     int((ancillas[i - 1] - (logical_qubit_number - 1) * (2 * distance + 1) ** 2) / (
                                             2 * distance + 1)), round))


def final_measurement(circuit, data_qubits, ancilla_X, ancilla_Z, distance, measurement_type, logical_qubit_number,
                      round, number_total_qubits=1):
    stabilizer_measurement(circuit, data_qubits, ancilla_Z, ancilla_X, distance, 0, 0, 0, measurement_type,
                           logical_qubit_number, round, number_total_qubits)
    circuit.append("TICK")
    if measurement_type == "Z":
        circuit.append("M", data_qubits)
        circuit.append_operation("OBSERVABLE_INCLUDE",
                                 [stim.target_rec(-len(data_qubits) + i) for i in range(0, distance)],
                                 logical_qubit_number - 1)
    if measurement_type == "X":
        circuit.append("MX", data_qubits)
        circuit.append_operation("OBSERVABLE_INCLUDE",
                                 [stim.target_rec(-len(data_qubits) + i) for i in range(0, distance ** 2, distance)],
                                 logical_qubit_number - 1)
    if measurement_type == "Y":
        circuit.append("MY", data_qubits)
        circuit.append_operation("OBSERVABLE_INCLUDE",
                                 [stim.target_rec(-len(data_qubits) + i) for i in range(0, distance ** 2)],
                                 logical_qubit_number - 1)


def surface_d_code_memory(circuit: stim.Circuit, distance, rounds, before_measure_flip_probability,
                          after_reset_flip_probability, after_clifford_polarisation, before_round_data_polarisation,
                          logical_qubit_number, initialization_type, measurement_type) -> stim.Circuit:
    circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z = surfacecode_init(circuit, distance, logical_qubit_number,
                                                                                initialization_type)
    for i in range(rounds):
        data_qubit_noise(circuit, data_qubits, before_round_data_polarisation)
        stabilizer_measurement(circuit, data_qubits, ancilla_qubits_Z, ancilla_qubits_X, distance,
                               before_measure_flip_probability, after_reset_flip_probability,
                               after_clifford_polarisation, initialization_type, logical_qubit_number, i)
    final_measurement(circuit, data_qubits, ancilla_qubits_X, ancilla_qubits_Z, distance, measurement_type,
                      logical_qubit_number, rounds)
    return circuit


def count_logical_errors_MWPM(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


if __name__ == "__main__":
    circuit = stim.Circuit()
    d = 3
    r = 2
    p = 0.01

    noisy_circuit = surface_d_code_memory(circuit=circuit,
                                          distance=d,
                                          rounds=d,
                                          before_measure_flip_probability=p,
                                          after_reset_flip_probability=0,
                                          after_clifford_polarisation=0,
                                          before_round_data_polarisation=p,
                                          logical_qubit_number=1,
                                          initialization_type="Z",
                                          measurement_type="Z")

    diagram = noisy_circuit.diagram('timeline-svg')
    with open("diagram_pheno.svg", 'w') as f:
        f.write(diagram.__str__())

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    Ls = [3, 5]  # , 7]
    for L in Ls:
        print("L = {}".format(L))
        num_shots = 200000

        # ps = np.arange(0., 0.06, 5e-3)
        ps = np.logspace(-3, -1, 10)

        ys = []

        for p in tqdm(ps):
            circuit = stim.Circuit()
            noisy_circuit = surface_d_code_memory(circuit=circuit,
                                                  distance=L,
                                                  rounds=L,
                                                  before_measure_flip_probability=p,
                                                  after_reset_flip_probability=0,
                                                  after_clifford_polarisation=0,
                                                  before_round_data_polarisation=p,
                                                  logical_qubit_number=1,
                                                  initialization_type="Z",
                                                  measurement_type="Z")
            # print(len(noisy_circuit.shortest_graphlike_error()))
            num_errors_sampled = count_logical_errors_MWPM(noisy_circuit, num_shots)

            ys.append(num_errors_sampled / num_shots)

        ys = np.array(ys)
        std_err = (ys * (1 - ys) / num_shots) ** 0.5
        ax.errorbar(ps, ys, std_err, fmt="-x", label="d=" + str(L))

    ax.axvline(x=0.029)
    ax.legend()
    ax.grid()
    ax.set_xlabel("$p$")
    ax.set_ylabel("$p_L$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.suptitle("Memory")
    plt.show()
