from abc import ABC, abstractmethod
from typing import Union

import stim
import numpy as np
from ..error_code.noise import NoiseModel


class QECCode(ABC):
    def __init__(self, distance, noise):
        self.distance = distance
        self.noise = noise
        if distance % 2 == 0:
            raise ValueError("Not optimal distance.")
        self.circuit = self.create_code_instance()

    def circuit_to_png(self):
        diagram = self.circuit.diagram('timeline-svg')
        with open("diagram_circuit-level.svg", 'w') as f:
            f.write(diagram.__str__())
        diagram = self.circuit.diagram('timeslice-svg')
        with open("diagram_slice.svg", 'w') as f:
            f.write(diagram.__str__())

    @abstractmethod
    def create_code_instance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_syndromes(self, n):
        raise NotImplementedError("Subclasses should implement this!")


class SurfaceCode(QECCode):
    def __init__(self, distance, noise: Union[float, NoiseModel], noise_model='circuit-level'):
        self.noise_model = noise_model
        super().__init__(distance, noise)

    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()

        list_pairs = [[1, -1], [+1, +1], [-1, +1], [-1, -1]]
        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

            circuit.append("TICK")

        return circuit

    def measure_all_x(self, coord_to_index, index_to_coordinate, list_x_ancillas_index, list_data_index):
        circuit = stim.Circuit()

        circuit.append("H", list_x_ancillas_index)
        # circuit.append("I", list_data_index)
        circuit.append("TICK")

        list_pairs = [[1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_x_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

            circuit.append("TICK")

        circuit.append("H", list_x_ancillas_index)
        # circuit.append("I", list_data_index)
        circuit.append("TICK")

        return circuit

    def measure_bell_stabilizers(self, coord_to_index, reference_qubit_index, reference_ancillas_index, Ly, Lx):
        circuit = stim.Circuit()

        # Z_R Z_L stabilizer
        circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[0]])
        for i in range(Ly):  # Ly):
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append("CNOT", [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[0]])
            circuit.append("TICK")

        # X_R X_L stabilizer
        circuit.append("H", reference_ancillas_index[1])  # 1 instead of Ly + 1
        circuit.append("TICK")
        circuit.append("CNOT", [reference_ancillas_index[1], reference_qubit_index])
        for i in range(Lx):  # Lx):
            for yi in range(Ly):
                y = 2 * yi + 1
                circuit.append("CNOT", [reference_ancillas_index[1], coord_to_index["({},{})".format(2 * i + 1, y)]])
        circuit.append("TICK")
        circuit.append("H", reference_ancillas_index[1])
        circuit.append("TICK")

        return circuit

    def create_code_instance(self):
        if self.noise_model == 'circuit-level':
            if not isinstance(self.noise, NoiseModel):
                self.noise = NoiseModel.CircuitLevel(p=self.noise)

            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance

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

            # print(coord_to_index)
            # print(index_to_coordinate)

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index, list_data_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

            circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
            circuit.append("TICK")

            # Measure Bell stabilizers
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
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
            stab_measure = self.noise.noisy_circuit(stab_measure)

            # Measure stabilizers N times
            circuit += stab_measure * (Ly - 1)  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

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

            # Measure Bell also one more time
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")
            circuit += measure_bell
            circuit.append("M", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
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


            '''
            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance

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

            for x in range(4, Lx_ancilla, 4):
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
                index_to_coordinate.append([x, 0])

                list_z_ancillas_index.append(qubit_idx)

                qubit_idx += 1

            for y in range(2, Ly_ancilla - 1, 2):
                yi = 2 - y % 4
                # print(yi)
                xs = np.arange(yi, 2 * Lx + yi // 2, 2)
                for dummy, x in enumerate(xs):
                    circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                    # coord_to_index[x,y] = qubit_idx
                    coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                    index_to_coordinate.append([x, y])

                    if dummy % 2 == 1:
                        list_z_ancillas_index.append(qubit_idx)
                    elif dummy % 2 == 0:
                        list_x_ancillas_index.append(qubit_idx)

                    qubit_idx += 1

            for x in range(2, Lx_ancilla - 1, 4):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
                index_to_coordinate.append([x, Ly_ancilla - 1])
                list_z_ancillas_index.append(qubit_idx)

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
            for i in range(1):  # Ly):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
                index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            # logical x reference qubit
            for i in range(1):  # Lx):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
                index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            # print(coord_to_index)
            # print(index_to_coordinate)

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

            circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
            circuit.append("TICK")

            # Measure Bell stabilizers
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            # X Measurement
            circuit += measure_x

            # Measure all X stabilizers

            circuit.append("M", list_x_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_x_ancillas_index)
            circuit.append("TICK")

            # For testing decoder
            circuit += measure_z

            # Measure all Z stabilizers

            circuit.append("M", list_z_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_z_ancillas_index)
            circuit.append("TICK")

            # Error correction cycles

            stab_measure = stim.Circuit()

            # Measure X
            stab_measure += measure_x
            stab_measure.append("M", list_x_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_x_ancillas_index)
            stab_measure.append("TICK")

            # Measure Z
            stab_measure += measure_z
            stab_measure.append("M", list_z_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_z_ancillas_index)
            stab_measure.append("TICK")

            stab_measure = self.noise.noisy_circuit(stab_measure)

            # Here changed for recurrent measurement
            # Measure logical operator
            stab_measure += measure_bell
            stab_measure.append("M", reference_ancillas)
            stab_measure.append("TICK")
            stab_measure.append("R", reference_ancillas)
            stab_measure.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * (offset + r_offset),
                        1 + idx + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset, 1 + idx + r_offset + offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))

            stab_measure.append_from_stim_program_text("SHIFT_COORDS(0,0,1)")

            # Measure stabilizers N times
            # Make stabilizer measurement (= QEC cycle) noisy
            circuit += stab_measure * Ly  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

            # Measure X noise-free
            circuit += measure_x
            circuit.append("M", list_x_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_x_ancillas_index)
            circuit.append("TICK")

            # Measure Z noise-free
            circuit += measure_z
            circuit.append("M", list_z_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_z_ancillas_index)
            circuit.append("TICK")

            # Measure Bell also d+1-th time
            circuit += measure_bell
            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * (offset + r_offset),
                        1 + idx + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset, 1 + idx + r_offset + offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))
            '''
            # Measure all data qubits

            # circuit.append("M", range(Lx * Ly))
            '''
            # Measure Bell stabilizers 2nd time
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
            offset = len(list_x_ancillas_index)

            # Measure Bell stabilizers
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                             1 + idx + r_offset + 2 * (
                                                                                                         Ly + 1) * offset))
            '''
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

            
            # Measure logical operator
            obs = ""
            for idx in range(1, Lx * Ly + 1): obs += " rec[-{}]".format(idx)
            circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)
            '''

            return circuit
        elif self.noise_model == 'phenomenological':
            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance

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

            # print(coord_to_index)
            # print(index_to_coordinate)

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index, list_data_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

            circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
            circuit.append("TICK")

            # Measure Bell stabilizers
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
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

            stab_measure.append("DEPOLARIZE1", list_data_index, self.noise)

            # Measure Z
            stab_measure += measure_z
            stab_measure.append("X_ERROR", list_z_ancillas_index, self.noise)
            stab_measure.append("M", list_z_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_z_ancillas_index)
            stab_measure.append("TICK")

            # Measure X
            stab_measure += measure_x
            stab_measure.append("X_ERROR", list_x_ancillas_index, self.noise)
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

            circuit.append("DEPOLARIZE1", list_data_index, self.noise)

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

            # Measure Bell also one more time
            circuit += measure_bell
            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
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
                #1 + idx + r_offset + 2 * (1 + 2) * offset, 1 + idx))
            ''' Circuit for several rounds
            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance

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

            # print(coord_to_index)
            # print(index_to_coordinate)

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

            circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
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

            # Measure Bell stabilizers
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            # Error correction cycles

            stab_measure = stim.Circuit()

            stab_measure.append("DEPOLARIZE1", list_data_index, self.noise)

            # Measure Z
            stab_measure += measure_z
            stab_measure.append("X_ERROR", list_z_ancillas_index, self.noise)
            stab_measure.append("M", list_z_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_z_ancillas_index)
            stab_measure.append("TICK")

            # Measure X
            stab_measure += measure_x
            stab_measure.append("X_ERROR", list_x_ancillas_index, self.noise)
            stab_measure.append("M", list_x_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_x_ancillas_index)
            stab_measure.append("TICK")

            # Here changed for recurrent measurement
            # Measure logical operator
            stab_measure += measure_bell
            stab_measure.append("M", reference_ancillas)
            stab_measure.append("TICK")
            stab_measure.append("R", reference_ancillas)
            stab_measure.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset,
                        1 + idx + offset + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * offset + 2 * r_offset, 1 + idx + r_offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))

            stab_measure.append_from_stim_program_text("SHIFT_COORDS(0,0,1)")

            # Measure stabilizers N times
            circuit += stab_measure * (Ly-1)  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

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

            # Measure Bell also d+1-th time
            circuit += measure_bell
            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset,
                        1 + idx + offset + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * offset + 2 * r_offset, 1 + idx + r_offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))
            '''
            # Measure all data qubits
            # circuit.append("M", range(Lx * Ly))

            # Here changed for recurrent measurements
            # Measure Bell stabilizers 2nd time

            '''
            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance

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

            for x in range(4, Lx_ancilla, 4):
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
                index_to_coordinate.append([x, 0])

                list_z_ancillas_index.append(qubit_idx)

                qubit_idx += 1

            for y in range(2, Ly_ancilla - 1, 2):
                yi = 2 - y % 4
                # print(yi)
                xs = np.arange(yi, 2 * Lx + yi // 2, 2)
                for dummy, x in enumerate(xs):
                    circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                    # coord_to_index[x,y] = qubit_idx
                    coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                    index_to_coordinate.append([x, y])

                    if dummy % 2 == 1:
                        list_z_ancillas_index.append(qubit_idx)
                    elif dummy % 2 == 0:
                        list_x_ancillas_index.append(qubit_idx)

                    qubit_idx += 1

            for x in range(2, Lx_ancilla - 1, 4):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
                # coord_to_index[x,y] = qubit_idx
                coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
                index_to_coordinate.append([x, Ly_ancilla - 1])
                list_z_ancillas_index.append(qubit_idx)

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
            for i in range(1):  # Ly):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
                index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            # logical x reference qubit
            for i in range(1):  # Lx):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
                index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            # print(coord_to_index)
            # print(index_to_coordinate)

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

            circuit.append("R", range(2 * Lx * Ly - 1 + 1 + len(reference_ancillas)))
            circuit.append("TICK")

            # Measure Bell stabilizers
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            # X Measurement
            circuit += measure_x

            # Measure all X stabilizers

            circuit.append("M", list_x_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_x_ancillas_index)
            circuit.append("TICK")

            # For testing decoder
            circuit += measure_z

            # Measure all Z stabilizers

            circuit.append("M", list_z_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_z_ancillas_index)
            circuit.append("TICK")

            # Error correction cycles

            stab_measure = stim.Circuit()

            stab_measure.append("DEPOLARIZE1", list_data_index, self.noise)

            # Measure X
            stab_measure += measure_x
            stab_measure.append("X_ERROR", list_x_ancillas_index, self.noise)
            stab_measure.append("M", list_x_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_x_ancillas_index)
            stab_measure.append("TICK")

            # Measure Z
            stab_measure += measure_z
            stab_measure.append("X_ERROR", list_z_ancillas_index, self.noise)
            stab_measure.append("M", list_z_ancillas_index)
            stab_measure.append("TICK")
            stab_measure.append("R", list_z_ancillas_index)
            stab_measure.append("TICK")

            # Here changed for recurrent measurement
            # Measure logical operator
            stab_measure += measure_bell
            stab_measure.append("M", reference_ancillas)
            stab_measure.append("TICK")
            stab_measure.append("R", reference_ancillas)
            stab_measure.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * (offset + r_offset),
                        1 + idx + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset, 1 + idx + r_offset + offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                stab_measure.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))

            stab_measure.append_from_stim_program_text("SHIFT_COORDS(0,0,1)")

            # Measure stabilizers N times
            circuit += stab_measure * (Ly - 1)  # here Ly since the first measurement doesn't belong to the QEC cycle and is noise-free

            # Measure X
            circuit += measure_x
            circuit.append("X_ERROR", list_x_ancillas_index, self.noise)
            circuit.append("M", list_x_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_x_ancillas_index)
            circuit.append("TICK")

            # Measure Z
            circuit += measure_z
            circuit.append("X_ERROR", list_z_ancillas_index, self.noise)
            circuit.append("M", list_z_ancillas_index)
            circuit.append("TICK")
            circuit.append("R", list_z_ancillas_index)
            circuit.append("TICK")

            # Here changed for recurrent measurement
            # Measure logical operator
            circuit += measure_bell
            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
            offset = (L ** 2 - 1) // 2

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 2 * (offset + r_offset),
                        1 + idx + r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{},0)".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + 3 * offset + 2 * r_offset, 1 + idx + r_offset + offset))

            # Here changed for recurrent measurement
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + r_offset + 2 * offset, 1 + idx))

            '''

            '''
            circuit += measure_bell

            circuit.append("M", reference_ancillas)
            circuit.append("TICK")
            circuit.append("R", reference_ancillas)
            circuit.append("TICK")

            r_offset = len(reference_ancillas)
            offset = len(list_x_ancillas_index)

            # Measure Bell stabilizers
            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                             1 + idx + r_offset + 2 * (Ly+1) * offset))
            '''
            return circuit
        else:
            # Code capacity setting
            if isinstance(self.noise, NoiseModel):
                raise AttributeError('Noise is not a float.')

            circuit = stim.Circuit()

            Lx, Ly = self.distance, self.distance
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
            for i in range(1):  # Ly):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
                index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            # logical x reference qubit
            for i in range(1):  # Lx):
                circuit.append_from_stim_program_text(
                    "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
                index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
                reference_ancillas.append(qubit_idx)
                qubit_idx += 1

            measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
            measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index, list_data_index)
            measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

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
            if self.noise_model == 'depolarizing':
                circuit.append("DEPOLARIZE1", list_data_index, self.noise)
            elif self.noise_model == 'bitflip':
                circuit.append("X_ERROR", list_data_index, self.noise)
            else:
                raise ValueError("Unknown noise_model {}".format(self.noise_model))
            circuit.append("TICK")

            circuit += measure_z
            circuit += measure_x
            circuit += measure_bell

            circuit.append("M", list_z_ancillas_index)
            circuit.append("M", list_x_ancillas_index)
            circuit.append("M", reference_ancillas)

            offset = (Lx * Ly - 1) // 2
            r_offset = len(reference_ancillas)

            for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(
                        1 + idx + offset + r_offset,
                        1 + idx + 3 * offset + 2 * r_offset))

            for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                             1 + idx + 2 * offset + 2 * r_offset))

            for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
                print('HALLO', coord_x, coord_y)
                circuit.append_from_stim_program_text(
                    "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                             1 + idx + r_offset + 2 * offset))

            return circuit

    def get_syndromes(self, n, only_syndromes: bool = False, every_round: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))

        assert not every_round

        '''
        reshaped_samples = samples.reshape(n, self.distance, -1)
        for j in np.arange(1, self.distance):
            reshaped_samples[:, j, -1] = reshaped_samples[:, j - 1, -1] ^ reshaped_samples[:, j, -1]
            reshaped_samples[:, j, -2] = reshaped_samples[:, j - 1, -2] ^ reshaped_samples[:, j, -2]
        
        samples = reshaped_samples.reshape(n, -1)
        if every_round:
            if only_syndromes:
                samples = reshaped_samples[:, :, :-2]
                return samples.reshape(n, -1)
            return samples
        else:
            logical = samples[:, -2:]
            syndromes = reshaped_samples[:, :, :-2].reshape(n, -1)
            samples = np.concatenate((syndromes, logical), axis=1)
            if only_syndromes:
                return samples[:, :-2]
            return samples
        '''
        if every_round:
            reshaped_samples = samples.reshape(n, self.distance + 1, -1)
            for j in np.arange(1, self.distance + 1):
                reshaped_samples[:, j, -1] = reshaped_samples[:, j - 1, -1] ^ reshaped_samples[:, j, -1]
                reshaped_samples[:, j, -2] = reshaped_samples[:, j - 1, -2] ^ reshaped_samples[:, j, -2]

            samples = reshaped_samples.reshape(n, -1)
            if only_syndromes:
                samples = reshaped_samples[:, :, :-2]
                return samples.reshape(n, -1)
        else:
            if only_syndromes:
                return samples[:, :-2]
        return samples


class RepetitionCode(QECCode):
    def __init__(self, distance, noise):
        super().__init__(distance, noise)

    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()
        list_pairs = [[+1, 0], [-1, 0]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

        circuit.append("TICK")

        return circuit

    def measure_all(self, coord_to_index, index_to_coordinate, list_data_index, list_log_ancillas_index):
        circuit = stim.Circuit()

        for ai in list_log_ancillas_index:
            for xi in list_data_index:
                circuit.append("CNOT", [xi, ai])

        circuit.append("TICK")

        return circuit

    def create_code_instance(self):
        circuit = stim.Circuit()

        L = self.distance
        L_ancilla = 2 * L

        coord_to_index = {}
        index_to_coordinate = []

        # data qubit coordinates
        qubit_idx = 0
        for i in range(L):
            j = 2 * i
            circuit.append_from_stim_program_text("QUBIT_COORDS({},0)".format(j) + " {}".format(qubit_idx))
            coord_to_index.update({"({},0)".format(j): qubit_idx})
            index_to_coordinate.append([j, 0])

            qubit_idx += 1

        # ancilla qubit coordinates
        list_z_ancillas_index = []
        list_data_index = []

        for i in range(L):
            list_data_index.append(i)

        for i in range(1, L + 1, 2):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},0)".format(i) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(i, 0): qubit_idx})
            index_to_coordinate.append([i, 0])

            list_z_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        # logical qubit coordinate
        list_log_ancilla_index = [qubit_idx]
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(L_ancilla, 0) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(L_ancilla, 0): qubit_idx})
        index_to_coordinate.append([L_ancilla, 0])
        qubit_idx += 1

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_logical = self.measure_all(coord_to_index, index_to_coordinate, list_data_index, list_log_ancilla_index)

        circuit.append("R", range(2 * L))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_logical

        circuit.append("MR", list_z_ancillas_index)
        circuit.append("MR", list_log_ancilla_index)
        circuit.append("TICK")

        stab_measure = stim.Circuit()

        # errors
        stab_measure.append("X_ERROR", list_data_index, self.noise)
        stab_measure.append("TICK")

        stab_measure += measure_z
        stab_measure += measure_logical

        stab_measure.append("X_ERROR", list_z_ancillas_index, self.noise)
        stab_measure.append("MR", list_z_ancillas_index)
        stab_measure.append("MR", list_log_ancilla_index)

        r_offset = len(list_log_ancilla_index)

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                         1 + idx + L - 1 + 2 * r_offset))

        for idx, ancilla_log_index in enumerate(list_log_ancilla_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_log_index]
            stab_measure.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                         1 + idx + L - 1 + r_offset))

        return circuit

    def get_syndromes(self, n, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))
        syndromes = samples
        if only_syndromes:
            return syndromes[:, :-1]  #  * self.distance]
        return syndromes


if __name__ == '__main__':
    np.set_printoptions(threshold=10_000)
    distance = 3
    # noise_model = 'phenomenological'
    noise_model = 'circuit-level'
    every_round = False
    # s = SurfaceCode(3, 0.01, 'circuit-level')
    s = SurfaceCode(distance, 0.02, noise_model=noise_model)
    circuit = s.circuit
    # print(circuit)
    syndromes = np.array(s.get_syndromes(10, every_round))
    print(syndromes.shape)
    if every_round:
        for i, syndrome in enumerate(syndromes):
            print(f'Syndrome {i}: ', np.array_split(syndrome, distance))
    else:
        for i, syndrome in enumerate(syndromes):
            print(f'Syndrome {i}: ')
            for d in range(distance):
                print(np.array_split(syndrome[:-2], distance)[d])
            print('Logical: ', syndrome[-2:])
    s.circuit_to_png()
    '''
    s0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 0])

    # s0 = np.zeros(6*24+2)
    # s0 = np.zeros(2)

    # s0 = np.array([0, 0, 0, 0, 0, 0, 0, 0,
    #                1, 0, 0, 0, 0, 0, 0, 0,
    #                0, 0, 0, 0, 0, 0, 0, 0,
    #                0, 0, 0, 0, 0, 0, 0, 0,
    #                0, 1])

    syndromes = np.array(s.get_syndromes(1000000, every_round))

    array_tuples = [tuple(arr[:-2]) for arr in syndromes]

    # Count occurrences
    counter = Counter(array_tuples)

    # Example: Check occurrences of [1,2,3]
    query = tuple(s0[:-2])
    print(counter[query])

    array_tuples = [tuple(arr) for arr in syndromes]

    # Count occurrences
    counter = Counter(array_tuples)

    # Example: Check occurrences of [1,2,3]
    query = tuple(s0)
    print(counter[query])
    '''
# Code from Luis:
# DATA QUBITS : 1,2,4,6,8,10,12,14,15

# ANCILLA QUBITS : 0,3,7,11,5,9,13,16
