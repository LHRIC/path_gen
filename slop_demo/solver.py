"""Module: solver (refactored v2 — vectorized)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class SolverConfig:
    tolerance: float = 1e-6
    max_iter: int = 100


def solver_func_1(points, cfg=SolverConfig()):
    # vectorized rewrite 1
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 1) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def solver_func_2(points, cfg=SolverConfig()):
    # vectorized rewrite 2
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 2) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def solver_func_3(points, cfg=SolverConfig()):
    # vectorized rewrite 3
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 3) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def solver_func_4(points, cfg=SolverConfig()):
    # vectorized rewrite 4
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 4) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def solver_func_5(points, cfg=SolverConfig()):
    # vectorized rewrite 5
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 5) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def solver_func_6(points, cfg=SolverConfig()):
    # vectorized rewrite 6
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 6) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)
