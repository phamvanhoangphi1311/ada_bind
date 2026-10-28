"""Minimal stand-in for the Drive `gear_ultimate` backend (tests only).

Implements just the contract pbind relies on: prepare_context / run_dev_cv_only /
clone_context_for_run / run_grid / audit_context, attach_evo, reporting helpers.
Features are random; the "trainer" is a logistic regression.
"""
__version__ = "fake-0"
GEAR_ULTIMATE_VERSION = __version__
