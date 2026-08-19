"""
Wrapper Python para o Expectimax otimizado em C.
Suporta iterative deepening com time budget.
"""

import ctypes
import os
import numpy as np

# Carrega a shared library
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "expectimax_c.so"))

# Configura tipos
_lib.init.argtypes = []
_lib.init.restype = None

_lib.select_action.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # grid
    ctypes.c_int,                   # time_budget_ms
    ctypes.c_int,                   # max_depth
]
_lib.select_action.restype = ctypes.c_int

_lib.select_action_info.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # grid
    ctypes.c_int,                   # time_budget_ms
    ctypes.c_int,                   # max_depth
    ctypes.POINTER(ctypes.c_int),  # result[2]
]
_lib.select_action_info.restype = None

# Inicializa
_lib.init()


class ExpectimaxAgent:
    def __init__(self, depth=10, time_budget_ms=100):
        self.depth = depth          # max depth (cap for iterative deepening)
        self.base_depth = depth
        self.time_budget_ms = time_budget_ms  # ms per move (0 = fixed depth)

    def select_action(self, grid):
        """Recebe numpy grid 4x4, retorna (ação, score)."""
        flat = grid.flatten().astype(ctypes.c_int)
        arr = (ctypes.c_int * 16)(*flat)
        result = (ctypes.c_int * 2)(0, 0)
        _lib.select_action_info(arr, self.time_budget_ms, self.depth, result)
        action = result[0]
        depth_reached = result[1]
        return action, float(depth_reached)
