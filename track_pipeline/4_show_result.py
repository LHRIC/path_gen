"""
Interactive viewer for a saved track_pipeline result (.npz written by
2_run_windowed.py / 3_run_global.py). Opens the map + radius plot in a
matplotlib window, with the raw geometric initial guess overlaid (dashed
grey) against the result (solid red) so you can see what the optimizer
changed.

Usage:  python 4_show_result.py path\to\result.npz
(or edit RESULT_PATH below and run with no args)
"""
import sys

import matplotlib.pyplot as plt

import pipeline_lib as pl

RESULT_PATH = "track_pipeline/outputs/CHANGE_ME.npz"  # edit, or pass as argv[1]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RESULT_PATH
    gate_file, gate_points, dofs = pl.load_result(path)
    print(f"Loaded {path} (gate file: {gate_file}, {len(gate_points)} gates)")
    guess_dofs = pl.so.initial_guess(gate_points)
    pl.plot_result(gate_points, dofs, path, guess_dofs=guess_dofs)
    plt.show()
