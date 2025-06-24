from enum import Enum, auto
from typing import Union
import numpy as np


class Space(Enum):
    BOOLEAN_CUBE = auto()      # {0,1}^n
    PLUS_MINUS_CUBE = auto()   # {-1,1}^n
    REAL = auto()              # ℝ^n
    LOG = auto()               # Log space
    GAUSSIAN = auto()          # Gaussian space

    @staticmethod
    def translate(input: Union[int, float, np.ndarray],
                  source_space: "Space",
                  target_space: "Space") -> Union[int, float, np.ndarray]:
        """
        Translate a scalar or array from one space to another.
        """
        if source_space == target_space:
            return input

        input = np.asarray(input)

        # Boolean (0/1) → ±1
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.PLUS_MINUS_CUBE:
            return 2 * input - 1

        # ±1 → Boolean (0/1)
        if source_space == Space.PLUS_MINUS_CUBE and target_space == Space.BOOLEAN_CUBE:
            return ((input + 1) // 2).astype(int)

        # Real-valued input → Boolean (mod 2)
        if source_space == Space.REAL and target_space == Space.BOOLEAN_CUBE:
            return (np.round(input) % 2).astype(int)

        # Boolean → Real (just cast)
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.REAL:
            return input.astype(float)

        # ±1 → Real
        if source_space == Space.PLUS_MINUS_CUBE and target_space == Space.REAL:
            return input.astype(float)

        # Real → ±1 (e.g., sign function)
        if source_space == Space.REAL and target_space == Space.PLUS_MINUS_CUBE:
            return np.where(input >= 0, 1, -1)

        # Stub: LOG, GAUSSIAN
        raise NotImplementedError(
            f"Translation from {source_space.name} to {target_space.name} not implemented."
        )
