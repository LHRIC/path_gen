"""Module: kinematics (refactored v2 — vectorized)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class KinematicsConfig:
    tolerance: float = 1e-6
    max_iter: int = 100


def kinematics_func_1(points, cfg=KinematicsConfig()):
    # vectorized rewrite 1
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 1) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def kinematics_func_2(points, cfg=KinematicsConfig()):
    # vectorized rewrite 2
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 2) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def kinematics_func_3(points, cfg=KinematicsConfig()):
    # vectorized rewrite 3
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 3) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)


def kinematics_func_4(points, cfg=KinematicsConfig()):
    # vectorized rewrite 4
    arr = np.asarray(points, dtype=float)
    out = np.tanh(arr * 4) - np.sqrt(np.abs(arr) + cfg.tolerance)
    return out.clip(-1.0, 1.0)
