"""Correctness-first Region-DiMO preparation utilities.

This package does not select a teacher and does not make the repository ready
for a formal DiMO run. See :mod:`src.dimo.contracts` for the hard gate.
"""

from .contracts import DIMO_EDIT_FORMAL_READY

__all__ = ["DIMO_EDIT_FORMAL_READY"]
