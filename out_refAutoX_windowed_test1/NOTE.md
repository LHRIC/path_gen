# ref_gates_autoX windowed spline

Method applied: same as the most recent endurance spline
(`track_pipeline/outputs/windowed_2026-08-23_15-19-39.csv`, from
`track_pipeline/2_run_windowed.py`) -- windowed sweep (window=6, step=5),
CONE_SPACING=0.30m, TRACK_WIDTH=1.22m, then `pipeline_lib.export_csv` at
0.5m spacing. No global refine step was run (the endurance reference didn't
have one at that point either).

`windowed_result.csv/.npz/.png` are reused, not freshly re-run: on
2026-08-23 15:28 the pipeline was already run with this exact method against
`autox_tracks/Michigan-2026-autox_gates.csv`, which is byte-identical to
the top-level `ref_gates_autoX.csv` (verified with `diff`, 52 gates). Rerunning
would have produced the same result, so the existing output was copied here
instead of spending optimizer time on a duplicate run.
