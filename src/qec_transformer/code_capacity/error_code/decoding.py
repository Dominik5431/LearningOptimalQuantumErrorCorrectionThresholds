from pathlib import Path

import numpy as np
import stim
import matplotlib.pyplot as plt
import pymatching
import torch

"""
This script is for testing the correct implementation of the surface codes.
It uses pymatching to find the MWPM-threshold of the surface code for distances 3, 5 and 7.
"""


def measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index):
    # Measure all z stabilizers

    circuit = stim.Circuit()
    list_pairs = [[1, -1], [-1, -1], [-1, +1], [+1, +1]]

    # order to measure z stabilizers
    # south-east, #south-west, north-west, north-east

    for xi, yi in list_pairs:

        for ancilla_qubit_idx in list_z_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                # print(ancilla_qubit_idx,data_qubit_idx)
                circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

            else:
                continue

        circuit.append("TICK")

    return circuit


def measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index):
    # Measure all X stabilizers

    circuit = stim.Circuit()

    circuit.append("H", list_x_ancillas_index)
    circuit.append("TICK")

    list_pairs = [[1, -1], [+1, +1], [-1, +1], [-1, -1]]

    # order to measure z stabilizers
    # south-east, north-east, north-west, #south-west

    for xi, yi in list_pairs:

        for ancilla_qubit_idx in list_x_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                # print(ancilla_qubit_idx,data_qubit_idx)
                circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

            else:
                continue

        circuit.append("TICK")

    circuit.append("H", list_x_ancillas_index)
    circuit.append("TICK")

    return circuit


def measure_bell_stabilizers(coord_to_index, reference_qubit_index, reference_ancillas_index, Ly, Lx):
    circuit = stim.Circuit()

    # Z_R Z_L stabilizer


    circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[0]])

    for i in range(Ly):  # Ly):
        circuit.append("TICK")
        for xi in range(Lx):
            x = 2 * xi + 1
            circuit.append("CNOT",
                           [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[0]])
            circuit.append("TICK")

    # X_R X_L stabilizer
    circuit.append("H", reference_ancillas_index[1])
    circuit.append("TICK")
    circuit.append("CNOT", [reference_ancillas_index[1], reference_qubit_index])

    for i in range(Lx):  # Lx):
        circuit.append("TICK")
        for yi in range(Ly):
            y = 2 * yi + 1
            circuit.append("CNOT",
                           [reference_ancillas_index[1], coord_to_index["({},{})".format(2 * i + 1, y)]])

            circuit.append("TICK")

    circuit.append("H", reference_ancillas_index[1])
    circuit.append("TICK")

    return circuit

def count_logical_errors(circuit: stim.Circuit, num_shots: int):
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    # detector_error_model = circuit.detector_error_model(decompose_errors=False)
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


def surface_code_capacity_zero(Lx, Ly, p):
    circuit = stim.Circuit()
    Lx_ancilla, Ly_ancilla = 2 * Lx + 1, 2 * Ly + 1

    coord_to_index = {}
    index_to_coordinate = []

    # data qubit coordinates
    qubit_idx = 0
    for yi in range(Ly):
        y = 2 * yi + 1
        for xi in range(Lx):
            x = 2 * xi + 1
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, y): qubit_idx})
            index_to_coordinate.append([x, y])

            qubit_idx += 1

    # ancilla qubit coordinates

    list_z_ancillas_index = []
    list_x_ancillas_index = []
    list_data_index = []

    for i in range(Lx * Ly):
        list_data_index.append(i)

    for x in range(2, Lx_ancilla - 1, 4):
        circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
        index_to_coordinate.append([x, 0])

        list_x_ancillas_index.append(qubit_idx)

        qubit_idx += 1

    for y in range(2, Ly_ancilla - 1, 2):
        yi = y % 4
        xs = range(yi, 2 * Lx + yi // 2, 2)
        for idx, x in enumerate(xs):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, y): qubit_idx})
            index_to_coordinate.append([x, y])

            if idx % 2 == 0:
                list_z_ancillas_index.append(qubit_idx)
            elif idx % 2 == 1:
                list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

    for x in range(4, Lx_ancilla, 4):
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
        index_to_coordinate.append([x, Ly_ancilla - 1])
        list_x_ancillas_index.append(qubit_idx)

        qubit_idx += 1

    # reference qubit coordinates
    reference_index = qubit_idx
    circuit.append_from_stim_program_text(
        "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1) + " {}".format(qubit_idx))
    coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1): qubit_idx})
    index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla - 1])
    qubit_idx += 1

    reference_ancillas = []
    # logical z reference qubit
    for i in range(1):
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
        index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
        reference_ancillas.append(qubit_idx)
        qubit_idx += 1

    # logical x reference qubit
    for i in range(1):
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
        index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
        reference_ancillas.append(qubit_idx)
        qubit_idx += 1

    measure_z = measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
    measure_x = measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
    measure_bell = measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

    circuit.append("R", range(2 * Lx * Ly - 1))
    circuit.append("TICK")

    circuit += measure_z
    circuit += measure_x
    circuit += measure_bell

    circuit.append("MR", list_z_ancillas_index)
    circuit.append("MR", list_x_ancillas_index)
    circuit.append("MR", reference_ancillas)
    circuit.append("TICK")

    # errors
    circuit.append("DEPOLARIZE1", list_data_index, p)
    circuit.append("TICK")

    circuit += measure_z
    circuit += measure_x
    circuit += measure_bell

    circuit.append("M", list_z_ancillas_index)
    circuit.append("M", list_x_ancillas_index)
    circuit.append("M", reference_ancillas)

    offset = (Lx * Ly - 1) // 2
    r_offset = len(reference_ancillas)

    msmt_schedule = []

    for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
        coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
        circuit.append_from_stim_program_text(
            "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + offset + r_offset,
                                                                                     1 + idx + 3 * offset + 2 * r_offset))
        # msmt_schedule.append(1 + idx + offset + r_offset)

    for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
        coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
        circuit.append_from_stim_program_text(
            "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                     1 + idx + 2 * offset + 2 * r_offset))
        # msmt_schedule.append(1 + idx + r_offset)

    for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
        # coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
        circuit.append_from_stim_program_text(
            "OBSERVABLE_INCLUDE({})".format(idx) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                     1 + idx + r_offset + 2 * offset))
        # msmt_schedule.append(1 + idx)

    # Measure all data qubits
    '''
    circuit.append("M", range(Lx * Ly))

    obs = ""

    for idx in range(1, Lx * Ly + 1): 
        obs += " rec[-{}]".format(idx)

    circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)
    '''
    return circuit


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    Ls = [3, 5, 7, 9]
    for L in Ls:
        print("L = {}".format(L))
        num_shots = 200000
        # ps = np.logspace(-3,-1,15)
        #ps = np.linspace(0.001, 0.2, 20)
        # ps = np.array([0])

        # ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.1))))
        ps = np.array([0.02, 0.05, 0.08, 0.11, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24])
        ys = []

        for p in ps:
            # noiseModel = NoiseModel.Code_Capacity(p=p)
            #noisy_circuit = surface_code_capacity_zero(L, L, p)
            # noiseModel = NoiseModel.CircuitLevel(p=p)
            # circuit= surface_code_circuit_level(L,L)
            # noisy_circuit = circuit= surface_code_circuit_level(L,L,p)
            # print(circuit)
            # print(circuit)
            # noisy_circuit = noiseModel.noisy_circuit(circuit)
            noisy_circuit = surface_code_capacity_zero(L, L, p)

            num_errors_sampled = count_logical_errors(noisy_circuit, num_shots)

            ys.append(num_errors_sampled / num_shots)

            # print(noisy_circuit)

        ys = np.array(ys)
        std_err = (ys * (1 - ys) / num_shots) ** 0.5
        # plt.plot(xs, ys,"-x", label="d=" + str(d))
        ax.errorbar(ps, ys, std_err, fmt="-x", label="d=" + str(L))

        # print(ys)
        parent_dir = Path(__file__).resolve().parent.parent.parent
        dict = {}
        for i, n in enumerate(ps):
            dict[n] = (torch.tensor(ys[i]), torch.tensor(std_err[i]), torch.tensor(std_err[i]))

        torch.save(dict,
                   "log_error_rate_{0}_{1}_mwpm.pt".format('ff16', L))

    ax.axvline(x=0.189)
    ax.legend()
    ax.grid()
    # ax.set_yscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel("$p_L$")
    plt.suptitle("Memory")
    # ax.loglog()
    # fig.savefig('plot.png')
    plt.show()
