import networkx as nx
import stim
import torch
from matplotlib import pyplot as plt


class SurfaceCodeCircuit:
    """
    Implements the surface code circuit for arbitrary distance.
    Possible implementations are available for bit-flip noise and depolarizing noise.
    """
    def __init__(self, distance, noise, noise_model='depolarizing'):
        self.noise_model = noise_model
        self.distance = distance
        self.noise = noise

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

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)

        circuit.append("R", range(2 * Lx * Ly - 1))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_x

        circuit.append("MR", list_z_ancillas_index)
        circuit.append("MR", list_x_ancillas_index)
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

        circuit.append("M", list_z_ancillas_index)
        circuit.append("M", list_x_ancillas_index)

        offset = (Lx * Ly - 1) // 2

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + offset,
                                                                                         1 + idx + 3 * offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                         1 + idx + 2 * offset))

        return circuit


class DEM_Graph:
    """
    Takes the surface code circuit, generates a detector-error model and builds an adjacency matrix
    that serves for the attention bias.
    """
    def __init__(self, distance, noise, noise_model):
        circuit = SurfaceCodeCircuit(distance, noise, noise_model).create_code_instance()
        self.distance = distance
        dem = circuit.detector_error_model()
        # print(dem)
        dem_text = str(dem)
        self.graph = nx.Graph()

        for line in dem_text.splitlines():
            if line.startswith("error"):
                parts = line.split()
                weight = float(parts[0].split('(')[1].rstrip(')'))  # Extract error probability
                error_nodes = [p for p in parts if p.startswith("D")]

                if len(error_nodes) == 1:
                    # Add a self-loop with the weight
                    self.graph.add_edge(error_nodes[0], error_nodes[0], weight=weight)
                else:
                    # Add weighted edges between detectors
                    for i in range(len(error_nodes)):
                        for j in range(i + 1, len(error_nodes)):
                            if self.graph.has_edge(error_nodes[i], error_nodes[j]):
                                # If the edge already exists, add the weight to the existing one
                                self.graph[error_nodes[i]][error_nodes[j]]['weight'] += weight
                            else:
                                self.graph.add_edge(error_nodes[i], error_nodes[j], weight=weight)

        # print(graph.edges())

    def get_graph(self):
        return self.graph

    def get_adjacency_matrix(self):
        node_order = []
        for i in range(self.distance ** 2 - 1):
            node_order.append('D{}'.format(i))
        return torch.as_tensor(nx.to_numpy_array(self.graph, nodelist=node_order, weight="weight"))


if __name__ == "__main__":
    # For testing
    graph = DEM_Graph(3, 0.03, 'depolarizing').get_graph()

    # Visualize the graph with weights
    pos = nx.spring_layout(graph)  # Compute layout positions
    edges = graph.edges(data=True)

    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold")
    edge_labels = {(u, v): f"{data['weight']:.3f}" for u, v, data in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red")

    plt.show()
