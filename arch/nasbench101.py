import copy
import json
from typing import Sequence, List, Any, Union, Set

import numpy as np

from utils.data_util import load_obj
from arch.graph import Graph


class NASBench101Graph(Graph):
    HASH = {'conv3x3-bn-relu': 0, 'conv1x1-bn-relu': 1, 'maxpool3x3': 2}
    HASH_T = {0: 'conv3x3-bn-relu', 1: 'conv1x1-bn-relu', 2: 'maxpool3x3'}

    def __init__(self, matrix: Union[np.ndarray, Sequence[Any]], ops: List[str]):
        super().__init__()
        self.original_matrix = copy.deepcopy(matrix) if matrix is np.ndarray else np.array(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(self.original_matrix)
        self.ops = copy.deepcopy(self.original_ops)
        self.valid_spec = True
        self.prune()
        self.fingerprint: str = self.hash_spec()

    @staticmethod
    def fill_graph(matrix: np.ndarray, ops: List[str], fill_nodes: int) -> tuple[np.ndarray, list[str]] | None:
        if len(matrix) == fill_nodes:
            return matrix, ops
        else:
            pad = fill_nodes - len(matrix)
            # new_matrix = np.pad(matrix, ((0, pad), (0, pad)), 'constant')
            new_matrix = np.zeros([fill_nodes, fill_nodes], dtype=matrix.dtype)
            new_matrix[:len(matrix) - 1, :len(matrix) - 1] = matrix[:-1, :-1]
            new_matrix[:len(matrix), -1] = matrix[:, -1]
            ops = ops[:-1] + np.random.choice(list(NASBench101Graph.HASH.keys()), pad).tolist() + ['output']
            return new_matrix, ops  # NASBench101Graph.fill_graph(matrix, ops, fill_nodes)

    def prune(self):
        """
        Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input: Set[int] = {0}
        frontier: List[int] = [0]
        while frontier:
            top: int = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = {num_vertices - 1}
        frontier = [num_vertices - 1]
        while frontier:
            top: int = frontier.pop()
            for v in range(top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous: set[int] = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def is_valid(self, module_vertices=7, max_edges=9):
        if not self.valid_spec:
            return False

        num_vertices = len(self.ops)
        num_edges = np.sum(self.matrix)

        if num_vertices > module_vertices:
            return False

        if num_edges > max_edges:
            return False

        if self.ops[0] != 'input':
            return False
        if self.ops[-1] != 'output':
            return False
        for op in self.ops[1:-1]:
            if op not in NASBench101Graph.HASH:
                return False
        return True

    def hash_spec(self):
        labeling = [-1] + [NASBench101Graph.HASH[op] for op in self.ops[1:-1]] + [-2]
        return self.hash(self.matrix, labeling)


class NASBench101_Helper:
    _instance = None

    num_vertices = 7
    max_edges = 9
    edge_spots = int(num_vertices * (num_vertices - 1) / 2)  # Upper triangular matrix
    edge_spots_idx = np.triu_indices(num_vertices, 1)
    op_spots = int(num_vertices - 2)  # Input/output vertices are fixed
    allowed_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
    allowed_edges = [0, 1]  # Binary adjacency matrix

    # upper and lower bound on the decision variables
    n_var = int(edge_spots + op_spots)
    lb = [0] * n_var
    ub = [1] * n_var
    ub[-op_spots:] = [2] * op_spots

    @classmethod
    def get_instance(cls):
        if NASBench101_Helper._instance is None:
            NASBench101_Helper._instance = NASBench101_Helper()
        return NASBench101_Helper._instance

    def __init__(self):
        if NASBench101_Helper._instance is None:
            print('### Start loading pkl file...')
            with open('config.json', 'r') as f:
                data = json.load(f)
            all_pkl_path = data['101_data']
            self.info = load_obj(all_pkl_path)
            print('### Loaded...')
        else:
            print('This class has been loaded.')

    @classmethod
    def encode(cls, arch: dict):
        # encode architecture phenotype to genotype
        # a sample arch = {'matrix': matrix, 'ops': ops}, where
        #     # Adjacency matrix of the module
        #     matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
        #             [0, 0, 0, 0, 0, 0, 1],  # op1
        #             [0, 0, 0, 0, 0, 0, 1],  # op2
        #             [0, 0, 0, 0, 1, 0, 0],  # op3
        #             [0, 0, 0, 0, 0, 0, 1],  # op4
        #             [0, 0, 0, 0, 0, 0, 1],  # op5
        #             [0, 0, 0, 0, 0, 0, 0]], # output layer
        #     # Operations at the vertices of the module, matches order of matrix
        #     ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
        x_edge = np.array(arch['matrix'])[NASBench101_Helper.edge_spots_idx]
        x_ops = np.empty(NASBench101_Helper.num_vertices - 2)
        for i, op in enumerate(arch['ops'][1:-1]):
            x_ops[i] = (np.array(NASBench101_Helper.allowed_ops) == op).nonzero()[0][0]
        return np.concatenate((x_edge, x_ops)).astype(int)

    @classmethod
    def decode(cls, x):
        x_edge = x[:NASBench101_Helper.edge_spots]
        x_ops = x[-NASBench101_Helper.op_spots:]
        matrix = np.zeros((NASBench101_Helper.num_vertices, NASBench101_Helper.num_vertices), dtype=int)
        matrix[NASBench101_Helper.edge_spots_idx] = x_edge
        ops = ['input'] + [NASBench101_Helper.allowed_ops[i] for i in x_ops] + ['output']
        return {'matrix': matrix, 'ops': ops}

    @classmethod
    def sample(cls, phenotype=True):
        matrix = np.random.choice(NASBench101_Helper.allowed_edges,
                                  size=(NASBench101_Helper.num_vertices, NASBench101_Helper.num_vertices))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(NASBench101_Helper.allowed_ops, size=NASBench101_Helper.num_vertices).tolist()
        ops[0] = 'input'
        ops[-1] = 'output'

        if phenotype:
            return {'matrix': matrix, 'ops': ops}
        else:
            return NASBench101_Helper.encode({'matrix': matrix, 'ops': ops})

    @classmethod
    def get_info(cls, _hash):
        return NASBench101_Helper.get_instance().info[_hash]
