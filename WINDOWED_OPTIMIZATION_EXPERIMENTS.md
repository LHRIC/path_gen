# Windowed optimization experiments (branch: windowed-local-optimization)

This branch is a **janky testing branch** — it's where the windowed/local
optimizer got rebuilt and tuned by trial and error. Nothing here is meant to
be the final, clean implementation; that's planned as a separate pass later
(see "Deferred work" at the bottom).

Test track for all of this: `ref_less_gates_twist.csv`, a 39-gate trouble
section (loop-prone, tight S-curves) cut from the gate-reduced 2019 Michigan
endurance track (`ref_less_gates.csv`, itself built from the user's
manually-simplified `endurance_tracks/Michigan-2019-endurance_gates.csv`).

## 1. Windowed optimizer redesign

`spline_optimize_windowed.py` sweeps a small window of gates along the track
instead of optimizing the whole thing at once (full-track optimization is
`spline_optimize_5.py`, and is much slower on big tracks).

**Old design:** each window's WINDOW_OVERLAP leading gates were frozen
(already solved by the previous window, never revisited), and only the
remaining gates were optimized. Gates got exactly one shot at being solved,
ever.

**New design (this branch):** every gate inside a window is optimized every
time a window touches it, including gates an earlier, overlapping window
already solved. WINDOW_STEP < WINDOW_SIZE means consecutive windows overlap
and re-optimize the shared gates with better information than before. One
gate before and one gate after the window are read in frozen, for the
objective function's context only (so the free gates have correct boundary
conditions), replacing the old design's asymmetric "whole overlap block
included in the objective" behavior, which wasted compute re-integrating
already-fixed segments for no benefit.

The last window of a sweep is pulled back to stay exactly WINDOW_SIZE gates
(rather than left short when the remaining gates don't divide evenly by
WINDOW_STEP), at the cost of a bigger overlap on that one final step.

## 2. Window/step experiments

Four configs tested on the 39-gate trouble section, all using the new design:

| config | time | full-track obj (lower=better) | length | windows |
|---|---|---|---|---|
| (a) window=10, step=9 | 500.2s | 1.930624 | 517.09 m | 5 |
| **(b) window=6, step=5** | **229.6s** | **1.927930** | 518.55 m | 8 |
| (c) window=8, step=4 | 496.4s | 1.936263 | 518.85 m | 9 |
| (d) window=6, step=3 | 250.4s | 1.939135 | 521.41 m | 12 |

**(b) window=6/step=5 wins on both axes** — fastest (2.2x faster than the
slowest) and best objective. All four comfortably beat the old frozen-overlap
design's best result (~2.02).

Counterintuitive finding: more overlap did not help. (d) (overlap=3, the most
re-optimization passes per gate) was both slower and slightly worse than (b)
(overlap=1). The "window greed" that heavier overlap was meant to mitigate
doesn't appear to be the dominant error source on this track — the extra
re-solves cost time without buying back quality.

Result files: `out_less_gates_twist_w10s9/`, `out_less_gates_twist_w6s5/`,
`out_less_gates_twist_w8s4/`, `out_less_gates_twist_w6s3/`.

## 3. Global optimizer comparison

`spline_optimize_5.py` (full-track, unconstrained D1/D2 bounds) warm-started
from the (d) w6s3 windowed result:

- **Runtime: 3286.147 s (54.8 min)**
- **Objective: 1.8705367590** — meaningfully better than any windowed config
  (~3% lower), as expected since it optimizes the whole track jointly instead
  of stitching together locally-optimal windows.
- Path length: 521.42 m.

Result files: `out_less_gates_twist_global_test1/`.

### D1/D2 gate-by-gate comparison (windowed vs. global)

Comparing each windowed config's D1/D2 DOFs against the global optimum
(absolute delta and delta/global-value) surfaced a few systematic blind
spots — gates where **all four windowed configs** disagree with global in
the same direction, regardless of window/step choice:

- **Gate 7 (D1x)** and **gate 29 (D1y)** are the clearest cases: every
  windowed config missed them the same way, by a consistent, non-trivial
  margin. These look like real geometric features windowing structurally
  can't reach without full-track context.
- Smaller same-direction misses at gates 5, 6, 8, 10, 12, 34.
- Gate 30's large single-run outlier (D1x/D1y off by 3.8-4.8 in the (d) run
  only) is noise specific to that run, not a shared pattern.

## 4. Bounded global optimizer experiment (not worth pursuing)

Tried tightening the global optimizer's D1/D2 bounds per-gate, anchored on
the windowed first-pass result (not the raw geometric guess — that
distinction mattered a lot, see below):

- D2 bound half-width = `max(0.5, 0.2 * |D2_windowed|)`
- D1 bound half-width = `max(1.5, 0.7 * |D1_windowed|)`
- Gate-position bounds unchanged from the original.
- Implemented as a separate script, `spline_optimize_5_bounded.py`, rather
  than modifying `spline_optimize_5.py` in place.

**Sanity check before running:** checked whether the *already-known* global
optimum even falls inside these bounds when anchored on each of the 4
windowed results. Anchored on the raw un-optimized geometric guess (a
mistaken first read of the requirement), 13-17% of D1/D2 components fell
outside the bounds -- including outright wrong-sign misses -- which would
have guaranteed a badly-degraded result. Anchored correctly on a windowed
first pass, the miss rate dropped to 0-6% with much smaller margins, meaning
the scheme was reasonable as actually specified. Anchor choice mattered:
(b) w6s5 had the fewest predicted misses (0/78 D1, 3/78 D2).

**Actual result** (bounded run, warm-started + anchored on the w6s3
windowed result), vs. the unconstrained global run above:

| | unconstrained global | bounded global | delta |
|---|---|---|---|
| objective (F) | 1.8705367590 | 1.9182520048 | +0.0477 (+2.55% worse) |
| runtime | 3286.147 s (54.8 min) | 2475.573 s (41.3 min) | -810.6 s (-24.7% faster) |
| accepted iterations | 95 | 56 | -39 (-41%) |
| total function evals | 45,080 | 37,044 | -17.8% |
| active bounds at end | 20 | 20 | same |

**Conclusion: not worth it.** Bounding cut runtime by about a quarter but
cost ~2.6% in path quality -- a mild, unfavorable speed/quality tradeoff, not
the 3-5x runtime cut that would have made it worthwhile. It didn't touch the
actual cost driver either: every iteration still finite-differences the
gradient over all ~195 DOFs regardless of how tight the box is (that's where
the ~62ms/objective-eval and 2079 rho_normalized calls per eval come from).
Tightening the box only cut the number of iterations needed to converge.

If a real 3-5x speedup is wanted later, the levers that would actually touch
per-iteration cost instead of iteration count:
- Analytic gradient instead of finite-difference (single biggest lever --
  real implementation work)
- Loosen `OBJ_FUN_INTEGRATION_EPSABS` further for this track size (currently
  1e-2)
- Reduce free-DOF count directly (e.g. hold D2 fixed for gates barely active
  in the windowed result), rather than narrowing the box around all of them

Decision: **runtime is annoying but acceptable as-is; this line of
experimentation is closed for now.**

## Deferred work (not started)

Noted during this branch's work, explicitly deferred to a later, cleaner
pass:
- Move all gate reference files into one folder.
- Add generation metadata (timestamps, source script/config) into the DOF
  output files themselves, to support a unified visualizer + lapsim export
  tool.

## Known loose ends on this branch

- `spline_optimize_5.py`'s `GUESS_DOF_FILE` is loaded via a bare relative
  path (unlike `FILENAME`, which resolves against the script's own
  directory) -- running it from the wrong working directory throws
  `FileNotFoundError`. Flagged, not fixed (pre-existing, out of scope for
  surgical changes).
- A separate, unrelated in-progress change was found sitting in this
  branch's working tree while committing this work: a staged rename of
  `ref_gates_endurance.csv` to `obs/ref_gates_endurance_obs_uneven_gates.csv`,
  plus a new, differently-sized `ref_gates_endurance.csv` at the original
  path. This wasn't produced by this line of work and wasn't touched by this
  commit -- it's left exactly as found for whoever owns that change to
  sort out.
