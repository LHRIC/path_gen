# Minimal reproduction of the import topology between the two real files.
# Mirrors: unpack_dofs imports spline_optimize_3 at top; the one shared function
# (load_dof_guess) is defined LATER in this file.
import opt_mod as so   # like 'import spline_optimize_3 as so'

X = 1  # stand-in for GATES_CSV block etc.

def load_dof_guess(path, expected_len=None):
    return ("loaded", path)

if __name__ == "__main__":
    print("unpack entry ran; so =", so.__name__)
