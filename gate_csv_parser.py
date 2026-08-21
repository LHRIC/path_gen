import numpy as np

def load_gates_2D(file_path):
    """
    Load gate cone positions from a CSV file (2D, no z).

    Columns in CSV: gate_index, x1, y1, z1, x2, y2, z2

    Returns numpy array of shape (N, 5): [gate_index, x1, y1, x2, y2]
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=float)
    return data[:, [0, 1, 2, 4, 5]]  # drop z1 (col 3) and z2 (col 6)


def load_gates_3D(file_path):
    """
    Load gate cone positions from a CSV file (3D, includes z).

    Columns in CSV: gate_index, x1, y1, z1, x2, y2, z2

    Returns numpy array of shape (N, 7): [gate_index, x1, y1, z1, x2, y2, z2]
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=float)
    return data