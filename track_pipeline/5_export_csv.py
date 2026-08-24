"""
Regenerate the evenly-spaced CSV at any spacing from an already-saved result
(.npz from 2_run_pipeline.py) -- fast, no re-optimization needed.

Usage:  python 4_export_csv.py path\to\result.npz 0.5
(or edit RESULT_PATH / SPACING below and run with no args)
"""
import sys

import pipeline_lib as pl

RESULT_PATH = "track_pipeline/outputs/CHANGE_ME.npz"  # edit, or pass as argv[1]
SPACING = 0.5  # meters, or pass as argv[2]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RESULT_PATH
    spacing = float(sys.argv[2]) if len(sys.argv) > 2 else SPACING
    gate_file, gate_points, dofs = pl.load_result(path)
    out_path = path.replace(".npz", f"_spacing{spacing}m.csv")
    pl.export_csv(gate_points, dofs, spacing, out_path)
