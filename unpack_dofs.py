"""
Unpack a saved DOF file (the opt_dofs_*.npy written by the report) and re-plot
it using the existing plot functions in spline_optimize_3, WITHOUT re-running the
optimizer and WITHOUT writing a report.

How it works:
  - spline_optimize_3 guards its optimize/report code behind __name__ == "__main__",
    so importing it here only brings in the functions and the gate setup
    (imported_gate_points, gate_ct, ...). No optimization runs on import.
  - The plot functions in spline_optimize_3 read module-level globals that normally
    come from the optimizer run (opt_dofs, x_spline, total_length, ...). We rebuild
    those exact globals from the loaded DOFs and inject them back into the module,
    then call the existing plot functions.

Typical use (one line at the top of spline_optimize_3 or any script):
    from unpack_dofs import plot_from_dof_file; plot_from_dof_file()

Calling plot_from_dof_file() with no path pops a file-explorer window so you can
pick which opt_dofs_*.npy to load.
"""

import numpy as np

import os

import spline_optimize_5 as so

# ----- Gate file used when unpacking (edit this) -----
# The DOF file you load must come from this same gate CSV. Setting this here
# overrides whatever FILENAME spline_optimize_3 was set to, so you don't have to
# edit the optimizer file just to re-plot a saved DOF export.
GATES_CSV = "ref_gates_gates.csv"
GATES_CSV = "ref_gates_2_gates.csv"
#GATES_CSV = "ref_gates_autoX.csv"
#GATES_CSV = "ref_gates_endurance.csv"
GATES_CSV = "ref_gates_enduranceTwist.csv"
# -----------------------------------------------------

# ----- Initial-guess export toggle (edit this) -----
# When True, unpack_dofs does NOT plot a loaded DOF file. Instead it builds the
# geometric initial guess for GATES_CSV (using spline_optimize_5.initial_guess),
# saves it to the "outputs" folder, and stops.
GENERATE_INITIAL_GUESS = False
OUTPUT_DIR = "outputs"
# ---------------------------------------------------


def load_gates():
    """Load the gate CSV named by GATES_CSV and push it onto the
    spline_optimize_3 module, overriding the gates it imported from its own
    FILENAME. The plot functions and DOF checks read so.imported_gate_points and
    so.gate_ct, so updating those two is what makes the unpacker use our CSV."""
    gate_file_path = os.path.join(so.current_path, GATES_CSV)
    so.imported_gate_points = so.load_gates_2D(gate_file_path)
    so.gate_ct = len(so.imported_gate_points[:, 0])
    so.FILENAME = GATES_CSV  # keep messages consistent with the file we loaded
    print(f"Using gate file: {GATES_CSV}  ({so.gate_ct} gates)")


def pick_dof_file():
    """Open a Windows file-explorer dialog and return the chosen .npy path.
    Returns None if the user cancels."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # hide the empty tk window, show only the file dialog
    path = filedialog.askopenfilename(
        title="Pick an exported DOF file to unpack",
        filetypes=[("DOF files", "*.npy"), ("All files", "*.*")],
        initialdir=so.current_path,
    )
    root.destroy()
    return path if path else None


def unpack_dofs(path):
    """Load a saved DOF file into the flat array generate_spline expects."""
    return np.load(path)


def load_dof_guess(path, expected_len=None):
    """Load a saved DOF file to use as a warm-start guess for the optimizer.

    This is the ONE function spline_optimize_3 imports. It is deliberately
    self-contained (only numpy, no reference to the spline_optimize_3 module) so
    that importing it does not create a circular import.

    path:         path to an opt_dofs_*.npy file.
    expected_len: if given (e.g. gate_ct * 5), the loaded array length is checked
                  against it and a clear error is raised on mismatch, since a
                  guess of the wrong length would crash generate_spline.
    Returns the flat DOF array.
    """
    dofs = np.load(path)
    if expected_len is not None and len(dofs) != expected_len:
        raise ValueError(
            f"\nGuess DOF file does not match the current gate file.\n"
            f"  '{path}' has {len(dofs)} values -> {len(dofs) / 5:.1f} gates.\n"
            f"  The current gate file expects {expected_len} values "
            f"({expected_len // 5} gates).\n"
            f"  Use a DOF file that was generated from the same gate CSV."
        )
    print(f"Loaded guess DOFs from: {path}  ({len(dofs)} values)")
    return dofs


def check_dof_count(opt_dofs):
    """Make sure the loaded DOF file matches the gate file currently set in
    spline_optimize_3 (FILENAME). A DOF array has 5 values per gate (1 gate
    position + 2 first-derivative + 2 second-derivative), so its length reveals
    the gate count it was generated for. If that doesn't match so.gate_ct, the
    splines would be built wrong (or crash), so stop with a clear message
    telling the user which gate file to switch to."""
    expected = so.gate_ct * 5
    actual = len(opt_dofs)
    if actual != expected:
        file_gates = actual / 5
        raise ValueError(
            f"\nDOF / gate-file mismatch.\n"
            f"  This DOF file has {actual} values -> {file_gates:.1f} gates.\n"
            f"  spline_optimize_3 is set to FILENAME = '{so.FILENAME}' "
            f"({so.gate_ct} gates, expects {expected} values).\n"
            f"  Fix: set FILENAME in spline_optimize_3.py to the gate file this "
            f"DOF export came from, then re-run."
        )


def print_dof_table(opt_dofs):
    """Print all DOFs in an evenly spaced table, one row per gate.
    Columns: gate position (0-1 across the gate), then the x/y first and second
    derivative DOFs. Matches the flat layout generate_spline expects:
      [pos_0..pos_(n-1), d1x..., d1y..., d2x..., d2y...]"""
    gate_ct = so.gate_ct
    pos = opt_dofs[:gate_ct]
    d1x = opt_dofs[gate_ct:2 * gate_ct]
    d1y = opt_dofs[2 * gate_ct:3 * gate_ct]
    d2x = opt_dofs[3 * gate_ct:4 * gate_ct]
    d2y = opt_dofs[4 * gate_ct:5 * gate_ct]

    headers = ["gate", "position", "D1x", "D1y", "D2x", "D2y"]
    width = 12  # fixed column width keeps everything evenly spaced
    header_row = "".join(h.rjust(width) for h in headers)
    print("\n" + header_row)
    print("-" * len(header_row))
    for i in range(gate_ct):
        cells = [str(i),
                 f"{pos[i]:.6f}",
                 f"{d1x[i]:.6f}", f"{d1y[i]:.6f}",
                 f"{d2x[i]:.6f}", f"{d2y[i]:.6f}"]
        print("".join(c.rjust(width) for c in cells))
    print()


def _inject_run_globals(opt_dofs):
    """Rebuild the optimizer-run globals from a DOF array and set them on the
    spline_optimize_3 module so its existing plot functions work. Mirrors the
    'Extract optimized results' block in spline_optimize_3."""
    gate_ct = so.gate_ct

    gate_x_pts, gate_y_pts = so.generate_gate_cross_points(so.imported_gate_points, opt_dofs)
    x_spline, y_spline = so.generate_spline(so.imported_gate_points, opt_dofs)
    total_length, distance_t = so.path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, gate_ct - 1, len(distance_t_full))
    equidistant_times = np.interp(np.linspace(0, total_length, so.PATHGEN_POINT_COUNT),
                                  distance_t_full, t_uniform)
    x_path_pts, y_path_pts = so.generate_path_t(x_spline, y_spline, equidistant_times)

    # The iteration-overlay plots need a path history. A loaded DOF file has none,
    # so seed it with this single loaded path (start == end). The overlay plot then
    # just shows the loaded path.
    iteration_paths = [(x_spline(np.linspace(0, gate_ct - 1, 300)),
                        y_spline(np.linspace(0, gate_ct - 1, 300)))]

    # Push everything the plot functions read back onto the module.
    so.opt_dofs = opt_dofs
    so.gate_x_pts, so.gate_y_pts = gate_x_pts, gate_y_pts
    so.x_spline, so.y_spline = x_spline, y_spline
    so.total_length, so.distance_t = total_length, distance_t
    so.distance_t_full = distance_t_full
    so.t_uniform = t_uniform
    so.equidistant_times = equidistant_times
    so.x_path_pts, so.y_path_pts = x_path_pts, y_path_pts
    so.iteration_paths = iteration_paths
    so.report_figures = []  # fresh list; we never write a report here


def plot_from_dof_file(path=None):
    """Load a DOF file (prompting via file explorer if no path is given), rebuild
    the splines, and show the existing map / radius / debug plots. No report is
    written. Returns the loaded DOF array, or None if the picker was cancelled."""
    load_gates()  # use GATES_CSV, not spline_optimize_3's FILENAME
    if path is None:
        path = pick_dof_file()
        if path is None:
            print("No file selected; nothing to plot.")
            return None

    print(f"Unpacking DOFs from: {path}")
    opt_dofs = unpack_dofs(path)
    check_dof_count(opt_dofs)  # stop early if the gate file doesn't match
    print_dof_table(opt_dofs)
    _inject_run_globals(opt_dofs)

    # Reuse the existing plot functions exactly as the optimizer does.
    so.plot_map_plots()
    so.plot_radius_over_distance()

    if so.PLOT_SPLINE_DEBUG:
        so.plot_spline_debug()
    if so.PLOT_PATH_DEBUG:
        so.plot_path_debug()

    return opt_dofs


def generate_and_save_initial_guess():
    """Build the geometric initial-guess DOFs for GATES_CSV and save them to the
    'outputs' folder as a flat .npy (the same layout generate_spline expects).
    The filename ends in GUESS_DOFS so it's easy to tell apart from optimized
    DOF exports. Returns the saved file path."""
    from datetime import datetime

    load_gates()  # use GATES_CSV; sets so.imported_gate_points / so.gate_ct
    guess_dofs = so.initial_guess(so.imported_gate_points)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(OUTPUT_DIR, f"opt_dofs_{stamp}_GUESS_DOFS.npy")
    np.save(out_path, guess_dofs)
    print(f"Saved initial-guess DOFs ({len(guess_dofs)} values) to: {out_path}")
    return out_path


if __name__ == "__main__":
    if GENERATE_INITIAL_GUESS:
        generate_and_save_initial_guess()
    else:
        plot_from_dof_file()
