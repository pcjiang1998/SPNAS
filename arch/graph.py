"""
Partly modified from https://github.com/google-research/nasbench
Apache License 2.0: https://github.com/google-research/nasbench/blob/master/LICENSE

Partly modified from https://github.com/EMI-Group/evoxbench
Apache License 2.0: https://github.com/EMI-Group/evoxbench/blob/main/LICENSE
"""
import hashlib
import itertools
from typing import Callable, Sequence, cast, List, AnyStr, Any, Tuple
import numpy as np


class Graph:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_edges(bits: int) -> Callable:
        return np.vectorize(
            lambda x, y: 0 if x >= y else (bits >> (x + (y * (y - 1) // 2))) % 2 == 1
        )

    @staticmethod
    def is_full(matrix: np.ndarray) -> bool:
        rows = np.any(np.all(matrix[: np.shape(matrix)[0] - 1, :] == 0, axis=1))
        cols = np.any(np.all(matrix[:, 1:] == 0, axis=0))
        return (not rows) and (not cols)

    @staticmethod
    def edges(matrix: np.ndarray) -> int:
        return cast(int, np.sum(matrix))

    @staticmethod
    def hash(matrix: np.ndarray, labels: List[int]) -> AnyStr:
        vertices: int = np.shape(matrix)[0]
        in_edges: List[int] = cast(list, np.sum(matrix, axis=0).tolist())
        out_edges: List[int] = cast(list, np.sum(matrix, axis=1).tolist())
        assert len(in_edges) == len(out_edges) == len(labels)
        hashes: List[AnyStr] = list(map(
            lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest(),
            list(zip(out_edges, in_edges, labels))
        ))
        for _ in range(vertices):
            _hashes: List[str] = []
            for v in range(vertices):
                in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
                _hashes.append(
                    hashlib.md5(
                        '|'.join((
                            ''.join(sorted(in_neighbors)),
                            ''.join(sorted(out_neighbors)),
                            hashes[v]
                        )).encode('utf-8')
                    ).hexdigest()
                )
            hashes = _hashes
        return hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    @staticmethod
    def permute(graph: np.ndarray, label: List[Any], permutation: Sequence[int]) -> Tuple[np.ndarray, List[Any]]:
        forward_perm = zip(permutation, list(range(len(permutation))))
        inverse_perm = [x[1] for x in sorted(forward_perm)]
        new_matrix = np.fromfunction(np.vectorize(lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1),
                                     (len(label), len(label)),
                                     dtype=np.int8)
        return cast(np.ndarray, new_matrix), [label[inverse_perm[i]] for i in range(len(label))]

    @staticmethod
    def is_isomorphic(graph_1: Any, graph_2: Any) -> bool:
        matrix_1, label_1 = np.array(graph_1[0]), graph_1[1]
        matrix_2, label_2 = np.array(graph_2[0]), graph_2[1]
        assert np.shape(matrix_1) == np.shape(matrix_2) and len(label_1) == len(label_2)
        vertices = np.shape(matrix_1)[0]
        for perm in itertools.permutations(range(vertices)):
            _matrix_1, _label_1 = Graph.permute(matrix_1, label_1, perm)
            if np.array_equal(_matrix_1, matrix_2) and _label_1 == label_2:
                return True
        return False

    @staticmethod
    def is_upper_triangular(matrix: np.ndarray) -> bool:
        """True if matrix is 0 on diagonal and below."""
        for i in range(np.shape(matrix)[0]):
            for j in range(0, i + 1):
                if matrix[i, j] != 0:
                    return False
        return True
