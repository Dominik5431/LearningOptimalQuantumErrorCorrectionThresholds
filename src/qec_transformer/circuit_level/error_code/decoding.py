import numpy as np
import stim
import matplotlib.pyplot as plt
import pymatching
from tqdm import tqdm

from ..error_code import NoiseModel

"""
This script is for testing the correct implementation of the surface codes.
It uses pymatching to find the MWPM-threshold of the surface code for distances 3, 5 and 7.
"""


def measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index):
    circuit = stim.Circuit()

    # list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]
    list_pairs = [[+1, -1], [+1, +1], [-1, -1], [-1, +1]]

    for xi, yi in list_pairs:
        for ancilla_qubit_idx in list_z_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

    circuit.append("TICK")

    return circuit


def measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index):
    circuit = stim.Circuit()

    circuit.append("H", list_x_ancillas_index)
    circuit.append("TICK")

    # list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

    list_pairs = [[+1, -1], [+1, +1], [-1, -1], [-1, +1]]

    for xi, yi in list_pairs:
        for ancilla_qubit_idx in list_x_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

    circuit.append("TICK")

    circuit.append("H", list_x_ancillas_index)
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


def surface_code_circuit(distance, noise, noise_model: str):
    """
        Logical zero code capacity
        :param distance: distance of the surface code
        :param noise: noise strength
        :return: stim circuit in the code_capacity setting
    """
    if noise_model == "depolarizing":
        circuit = stim.Circuit()
        # initialize qubits in |0> state
        # data qubits
        for n in np.arange(2 * distance ** 2):
            circuit.append("R", [n])
        # stabilizer qubits
        for i in np.arange(2 * distance ** 2 - 2):
            circuit.append("R", [2 * distance ** 2 + i])
        circuit.append("TICK")
        # Encoding
        # Measure all stabilizers to project into eigenstate of stabilizers, use stim's detector annotation
        # Z stabilizers
        for i in np.arange(distance):
            for j in np.arange(distance):
                # last stabilizer -> not applied due to constraint
                if i * distance + j == distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * distance + j, 2 * distance ** 2 + i * distance + j])
                if j % distance == distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                else:
                    circuit.append("CX", [i * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                    # vertical CNOTs
                circuit.append("CX", [distance ** 2 + i * distance + j,
                                      2 * distance ** 2 + i * distance + j])
                if i % distance == 0:
                    circuit.append("CX", [distance ** 2 + (i - 1) * distance + j + distance ** 2,
                                          2 * distance ** 2 + i * distance + j])
                else:
                    circuit.append("CX", [distance ** 2 + (i - 1) * distance + j,
                                          2 * distance ** 2 + i * distance + j])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                           2 * distance ** 2 + 2 * distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(distance):
            for j in np.arange(distance):
                # horizontal CNOTs
                if i * distance + j == distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + i * distance + j])
                if j % distance == 0:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          distance ** 2 + (i + 1) * distance + j - 1])
                else:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          distance ** 2 + i * distance + j - 1])
                    # vertical CNOTs
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      i * distance + j])
                if i % distance == distance - 1:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          (i + 1) * distance + j - distance ** 2])
                else:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          (i + 1) * distance + j])
            # Hadamard gates
        for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                           2 * distance ** 2 + 2 * distance ** 2 - 2):
            circuit.append("H", [i])
            # Measurement of syndrome qubits
        for i in np.arange(2 * distance ** 2 - 2):
            circuit.append("MR", [2 * distance ** 2 + i])
        circuit.append("TICK")
        # Noise
        for i in np.arange(2 * distance ** 2):
            circuit.append("X_ERROR", [i], noise)
        circuit.append("TICK")
        # Measure all stabilizers again:
        # Z stabilizers
        for i in np.arange(distance):
            for j in np.arange(distance):
                # last stabilizer -> not applied due to constraint
                if i * distance + j == distance ** 2 - 1:
                    break
                    # horizontal CNOTs
                circuit.append("CX", [i * distance + j, 2 * distance ** 2 + i * distance + j])
                if j % distance == distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                else:
                    circuit.append("CX", [i * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                    # vertical CNOTs
                circuit.append("CX", [distance ** 2 + i * distance + j,
                                      2 * distance ** 2 + i * distance + j])
                if i % distance == 0:
                    circuit.append("CX", [distance ** 2 + (i - 1) * distance + j + distance ** 2,
                                          2 * distance ** 2 + i * distance + j])
                else:
                    circuit.append("CX", [distance ** 2 + (i - 1) * distance + j,
                                          2 * distance ** 2 + i * distance + j])
            # X stabilizers
            # Hadamard gates
        for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                           2 * distance ** 2 + 2 * distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(distance):
            for j in np.arange(distance):
                # horizontal CNOTs
                if i * distance + j == distance ** 2 - 1:
                    break
                    # horizontal CNOTs
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + i * distance + j])
                if j % distance == 0:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          distance ** 2 + (i + 1) * distance + j - 1])
                else:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          distance ** 2 + i * distance + j - 1])
                    # vertical CNOTs
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      i * distance + j])
                if i % distance == distance - 1:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          (i + 1) * distance + j - distance ** 2])
                else:
                    circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                          (i + 1) * distance + j])
            # Hadamard gates
        for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                           2 * distance ** 2 + 2 * distance ** 2 - 2):
            circuit.append("H", [i])
            # Measurement of syndrome qubits
        for i in np.arange(2 * distance ** 2 - 2):
            circuit.append("MR", [2 * distance ** 2 + i])
            # Add detectors
        for i in np.arange(distance ** 2 - 1):  # changed something here
            circuit.append_from_stim_program_text("DETECTOR({},{})".format(i // distance, i % distance) + " rec[{}] rec[{}]".format(-2 * distance ** 2 + 2 - 2 * distance ** 2 + 2 + i, -2 * distance ** 2 + 2 + i))
        for i in np.arange(2 * distance ** 2):
            circuit.append("M", [i])
        obs = ""
        for idx in range(1, 2 * distance ** 2 + 1):
            obs += " rec[-{}]".format(idx)
        circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)
        return circuit
    elif noise_model == 'phenomenological':
        circuit = stim.Circuit()

        Lx, Ly = distance, distance

        Lx_ancilla = 2 * Lx + 1
        Ly_ancilla = 2 * Ly + 1
        # coord_to_index = np.zeros((Lx_ancilla,Ly_ancilla),dtype=int)
        coord_to_index = {}
        index_to_coordinate = []
        L = Lx
        # Data qubit coordinates
        qubit_idx = 0
        for yi in range(Ly):
            y = 2 * yi + 1
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                qubit_idx += 1

        # Ancilla qubit coordinates

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

        # print(coord_to_index)
        # print(index_to_coordinate)

        measure_z = measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)

        circuit.append("R", range(2 * Lx * Ly - 1))
        circuit.append("TICK")

        # For testing decoder
        circuit += measure_z

        # Measure all Z stabilizers

        circuit.append("M", list_z_ancillas_index)
        circuit.append("TICK")
        circuit.append("R", list_z_ancillas_index)
        circuit.append("TICK")

        # X Measurement
        circuit += measure_x

        # Measure all X stabilizers

        circuit.append("M", list_x_ancillas_index)
        circuit.append("TICK")
        circuit.append("R", list_x_ancillas_index)
        circuit.append("TICK")

        # Error correction cycles

        stab_measure = stim.Circuit()

        stab_measure.append("DEPOLARIZE1", list_data_index, noise)

        # Measure Z
        stab_measure += measure_z
        stab_measure.append("X_ERROR", list_z_ancillas_index, noise)
        stab_measure.append("M", list_z_ancillas_index)
        stab_measure.append("TICK")
        stab_measure.append("R", list_z_ancillas_index)
        stab_measure.append("TICK")

        # Measure X
        stab_measure += measure_x
        stab_measure.append("X_ERROR", list_x_ancillas_index, noise)
        stab_measure.append("M", list_x_ancillas_index)
        stab_measure.append("TICK")
        stab_measure.append("R", list_x_ancillas_index)
        stab_measure.append("TICK")

        offset = (L ** 2 - 1) // 2

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 3 * offset,
                    1 + idx + offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 2 * offset, 1 + idx))

        stab_measure.append_from_stim_program_text("SHIFT_COORDS(0,0,1)")

        # Measure stabilizers N times
        circuit += stab_measure * (Ly - 1)  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

        # circuit.append("DEPOLARIZE1", list_data_index, noise)

        # Measure Z noise-free
        circuit += measure_z
        circuit.append("M", list_z_ancillas_index)
        circuit.append("TICK")
        circuit.append("R", list_z_ancillas_index)
        circuit.append("TICK")

        # Measure X noise-free
        circuit += measure_x
        circuit.append("M", list_x_ancillas_index)
        circuit.append("TICK")
        circuit.append("R", list_x_ancillas_index)
        circuit.append("TICK")

        r_offset = 0
        offset = (L ** 2 - 1) // 2

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 3 * offset + r_offset,
                    1 + idx + offset + r_offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 2 * offset + r_offset, 1 + idx + r_offset))


        # Measure all data qubits
        circuit.append("M", range(Lx * Ly))

        # Measure stabilizers

        list_pairs = [[1, -1], [-1, -1], [-1, +1], [+1, +1]]
        all_data_qubits = list(range(Lx * Ly))
        '''
        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):

            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            dataqubits = []
            for xi, yi in list_pairs:
                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    dataqubits.append(data_qubit_idx)

            stab = ""
            for dq in dataqubits:
                stab += " rec[-{}]".format(Lx * Ly - dq)

            circuit.append_from_stim_program_text("DETECTOR({},{},{})".format(coord_x, coord_y, L - 1) + stab)
        '''
        # Measure stabilizers
        '''
        list_pairs = [[1, -1], [-1, -1], [-1, +1], [+1, +1]]
        all_data_qubits = list(range(Lx * Ly))

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):

            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            dataqubits = []
            for xi, yi in list_pairs:
                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    dataqubits.append(data_qubit_idx)

            stab = ""
            for dq in dataqubits:
                stab += " rec[-{}]".format(Lx * Ly - dq)

            circuit.append_from_stim_program_text("DETECTOR({},{},{})".format(coord_x, coord_y, L - 1) + stab)
        '''
        # Measure logical operator
        obs = ""
        for idx in range(1, L ** 2 + 1):
            obs += " rec[-{}]".format(idx)
        circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)

        return circuit
    elif noise_model == 'circuit-level':
        if not isinstance(noise, NoiseModel):
            noise = NoiseModel.CircuitLevel(p=noise)

        circuit = stim.Circuit()

        Lx, Ly = distance, distance

        Lx_ancilla = 2 * Lx + 1
        Ly_ancilla = 2 * Ly + 1
        # coord_to_index = np.zeros((Lx_ancilla,Ly_ancilla),dtype=int)
        coord_to_index = {}
        index_to_coordinate = []
        L = Lx
        # Data qubit coordinates
        qubit_idx = 0
        for yi in range(Ly):
            y = 2 * yi + 1
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                qubit_idx += 1

        # Ancilla qubit coordinates

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

        # print(coord_to_index)
        # print(index_to_coordinate)

        measure_z = measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index, list_data_index)

        circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
        circuit.append("TICK")

        # For testing decoder
        circuit += measure_z

        # Measure all Z stabilizers
        circuit.append("M", list_z_ancillas_index)
        circuit.append("TICK")

        # X Measurement
        circuit += measure_x

        # Measure all X stabilizers
        circuit.append("M", list_x_ancillas_index)
        circuit.append("TICK")

        # Error correction cycles
        stab_measure = stim.Circuit()

        # Measure Z
        stab_measure.append("R", list_z_ancillas_index)
        stab_measure.append("I", list_data_index)
        stab_measure.append("TICK")
        stab_measure += measure_z
        stab_measure.append("M", list_z_ancillas_index)
        stab_measure.append("I", list_data_index)
        stab_measure.append("TICK")

        # Measure X
        stab_measure.append("R", list_x_ancillas_index)
        stab_measure.append("I", list_data_index)
        stab_measure.append("TICK")
        stab_measure += measure_x
        stab_measure.append("M", list_x_ancillas_index)
        stab_measure.append("I", list_data_index)
        stab_measure.append("TICK")

        offset = (L ** 2 - 1) // 2

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 3 * offset,
                    1 + idx + offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 2 * offset, 1 + idx))

        stab_measure.append_from_stim_program_text("SHIFT_COORDS(0,0,1)")
        stab_measure = noise.noisy_circuit(stab_measure)

        # Measure stabilizers N times
        circuit += stab_measure * (
                    Ly - 1)  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

        # Measure Z noise-free
        circuit.append("R", list_z_ancillas_index)
        circuit.append("TICK")
        circuit += measure_z
        circuit.append("M", list_z_ancillas_index)
        circuit.append("TICK")

        # Measure X noise-free
        circuit.append("R", list_x_ancillas_index)
        circuit.append("TICK")
        circuit += measure_x
        circuit.append("M", list_x_ancillas_index)
        circuit.append("TICK")

        r_offset = 0
        offset = (L ** 2 - 1) // 2

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 3 * offset + r_offset,
                    1 + idx + offset + r_offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + 2 * offset + r_offset, 1 + idx + r_offset))

        # Here changed for recurrent measurement
        for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                    1 + idx + r_offset + 2 * (Ly - 1 + 2) * offset, 1 + idx))

        # Measure logical operator
        obs = ""
        for idx in range(1, L ** 2 + 1):
            obs += " rec[-{}]".format(idx)
        circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)

        return circuit


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    circuit = surface_code_circuit(distance=3, noise=0.03, noise_model='phenomenological')
    diagram = circuit.diagram('timeslice-svg')
    with open("diagram_testing.svg", 'w') as f:
        f.write(diagram.__str__())
    print(circuit)

    Ls = [3, 5, 7]
    for L in Ls:
        print("L = {}".format(L))
        num_shots = 200000
        # ps = np.logspace(-3,-1,15)
        # ps = np.linspace(0.001, 0.2, 20)
        # ps = np.array([0])

        # ps = np.arange(0., 0.03, 1e-3)
        ps = np.logspace(-2, -1, 20)

        ys = []

        for p in tqdm(ps):
            # noiseModel = NoiseModel.Code_Capacity(p=p)
            # noisy_circuit = surface_code_capacity_zero(L, L, p)
            # noiseModel = NoiseModel.CircuitLevel(p=p)
            # circuit= surface_code_circuit_level(L,L)
            # noisy_circuit = circuit= surface_code_circuit_level(L,L,p)
            # print(circuit)
            # print(circuit)
            # noisy_circuit = noiseModel.noisy_circuit(circuit)
            noisy_circuit = surface_code_circuit(distance=L, noise=p, noise_model='phenomenological')
            # print(len(noisy_circuit.shortest_graphlike_error()))

            num_errors_sampled = count_logical_errors(noisy_circuit, num_shots)

            ys.append(num_errors_sampled / num_shots)

            # print(noisy_circuit)

        ys = np.array(ys)
        std_err = (ys * (1 - ys) / num_shots) ** 0.5
        # plt.plot(xs, ys,"-x", label="d=" + str(d))
        ax.errorbar(ps, ys, std_err, fmt="-x", label="d=" + str(L))
        # ax.plot(ps, ys, label="d=" + str(L))
        # print(ys)

    ax.axvline(x=0.06)
    ax.axvline(x=0.043)
    ax.legend()
    ax.grid()
    # ax.set_yscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel("$p_L$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.suptitle("Memory")
    # ax.loglog()
    # fig.savefig('plot.png')
    plt.show()
