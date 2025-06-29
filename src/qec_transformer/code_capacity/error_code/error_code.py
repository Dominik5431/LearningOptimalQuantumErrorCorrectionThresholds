from abc import ABC, abstractmethod
import stim
import numpy as np


class QECCode(ABC):
    """
    Abstract base class for error codes.
    Defines the basis methods create_code_instance() and get_syndromes(n) with n: number of samples.
    Implements circuit_to_png() to visualize the error correcting circuit.
    """

    def __init__(self, distance, noise):
        self.distance = distance
        self.noise = noise
        if distance % 2 == 0:
            raise ValueError("Not optimal distance.")
        self.circuit = self.create_code_instance()

    def circuit_to_png(self):
        diagram = self.circuit.diagram('timeline-svg')
        with open("diagram_testing.svg", 'w') as f:
            f.write(diagram.__str__())

    @abstractmethod
    def create_code_instance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_syndromes(self, n):
        raise NotImplementedError("Subclasses should implement this!")


class SurfaceCode(QECCode):
    """
    Implementation of the Surface code in the code capacity setting including the measurement of logical operators
    along the syndromes.
    Available values for noise_model: depolarizing and bitflip.
    Available values for logical: maximal and string
    """

    def __init__(self, distance, noise, noise_model='depolarizing', logical='maximal'):
        self.noise_model = noise_model
        self.logical = logical

        if self.logical != 'maximal' and self.logical != 'string':
            raise ValueError("Only 'maximal' or 'string' logical logical are allowed.")
        super().__init__(distance, noise)

    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()
        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

        circuit.append("TICK")

        return circuit

    def measure_all_x(self, coord_to_index, index_to_coordinate, list_x_ancillas_index):
        circuit = stim.Circuit()

        circuit.append("H", list_x_ancillas_index)
        circuit.append("TICK")

        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

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

    def measure_bell_stabilizers(self, coord_to_index, reference_qubit_index, reference_ancillas_index, Ly, Lx):
        circuit = stim.Circuit()

        # Z_R Z_L stabilizer

        if self.logical == 'maximal':
            circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[0]])

        for i in range(Ly):  # Ly):
            if self.logical == 'string':
                circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[i]])
            circuit.append("TICK")
            for xi in range(Lx):
                x = 2 * xi + 1
                if self.logical == 'maximal':
                    circuit.append("CNOT",
                                   [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[0]])
                elif self.logical == 'string':
                    circuit.append("CNOT",
                                   [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[i]])
            circuit.append("TICK")

        # X_R X_L stabilizer

        if self.logical == 'maximal':
            circuit.append("H", reference_ancillas_index[1])
            circuit.append("TICK")
            circuit.append("CNOT", [reference_ancillas_index[1], reference_qubit_index])

        for i in range(Lx):  # Lx):
            if self.logical == 'string':
                circuit.append("H", reference_ancillas_index[Ly + i])
                circuit.append("TICK")
                circuit.append("CNOT", [reference_ancillas_index[Ly + i], reference_qubit_index])
            circuit.append("TICK")
            for yi in range(Ly):
                y = 2 * yi + 1
                if self.logical == 'maximal':
                    circuit.append("CNOT",
                                   [reference_ancillas_index[1], coord_to_index["({},{})".format(2 * i + 1, y)]])
                elif self.logical == 'string':
                    circuit.append("CNOT",
                                   [reference_ancillas_index[Ly + i], coord_to_index["({},{})".format(2 * i + 1, y)]])
                circuit.append("TICK")
            if self.logical == 'string':
                circuit.append("H", reference_ancillas_index[Ly + i])
                circuit.append("TICK")

        if self.logical == 'maximal':
            circuit.append("H", reference_ancillas_index[1])
            circuit.append("TICK")

        return circuit

    def create_code_instance(self):
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
        for i in range(Ly if self.logical == 'string' else 1):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
            index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        # logical x reference qubit
        for i in range(Lx if self.logical == 'string' else 1):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
            index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
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

        msmt_schedule = []

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + offset + r_offset,
                                                                                         1 + idx + 3 * offset + 2 * r_offset))
            msmt_schedule.append(1 + idx + offset + r_offset)

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                         1 + idx + 2 * offset + 2 * r_offset))
            msmt_schedule.append(1 + idx + r_offset)

        for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                         1 + idx + r_offset + 2 * offset))
            msmt_schedule.append(1 + idx)

        # Include all measurements
        rounds = 2
        for i in np.arange(rounds):
            for j, idx in enumerate(msmt_schedule):
                circuit.append_from_stim_program_text(
                    f"OBSERVABLE_INCLUDE({j + i * (2 * offset + r_offset)})" + f" rec[-{idx + (rounds - 1 - i) * (2 * offset + r_offset)}]")

        return circuit

    def get_syndromes(self, n, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples, observables = sampler.sample(shots=n, separate_observables=True)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))
        measurements = np.array(list(map(lambda y: np.where(y, 1, 0), observables)))

        rounds = np.size(measurements, 1) // np.size(samples, 1)
        measurement_rounds = np.array_split(measurements, rounds, axis=1)
        measurement_rounds = [np.expand_dims(arr, axis=-1) for arr in measurement_rounds]
        measurements_reshaped = np.concatenate(measurement_rounds, axis=-1)

        if only_syndromes:
            samples = samples[:, :-2 * (self.distance if self.logical == 'string' else 1)]
            measurements = measurements_reshaped[:, :self.distance ** 2 - 1, :]
            samples_exp = np.expand_dims(samples, axis=-1)
            return np.concatenate((samples_exp, measurements), axis=-1)

        syndromes_exp = np.expand_dims(samples, axis=-1)
        # measurements_reshaped = np.reshape(measurements,(np.size(syndromes, 0), np.size(syndromes, 1), rounds))
        return np.concatenate((syndromes_exp, measurements_reshaped), axis=-1)


class RepetitionCode(QECCode):
    """
    Implementation of the Repetition code in the code capacity setting including the measurement of logical operators
    along the syndromes.
    The repetition code is suitable to detect and correct bit-flip errors with a high threshold.
    """

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

        # errors
        circuit.append("X_ERROR", list_data_index, self.noise)
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_logical

        circuit.append("M", list_z_ancillas_index)
        circuit.append("M", list_log_ancilla_index)

        r_offset = len(list_log_ancilla_index)

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                         1 + idx + L - 1 + 2 * r_offset))

        for idx, ancilla_log_index in enumerate(list_log_ancilla_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_log_index]
            circuit.append_from_stim_program_text(
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
    # For testing:
    np.set_printoptions(threshold=10_000)
    s = SurfaceCode(9, 0.02, noise_model='depolarizing')
    syndromes = s.get_syndromes(10)
    events = syndromes[:, :, 0]
    msmts = syndromes[:, :, 1:]
    ms1, ms2 = msmts[:, :, 0], msmts[:, :, 1]
    print(syndromes[:, :, 0])
    assert (ms1 ^ ms2 == events).all()
    # s.circuit_to_png()
