"""Controlled information availability before WRITE, with byte invariants."""
from __future__ import annotations

from .runner import run_episode
from ..artifacts import canonical


def run_interventions(episode, policy_factory, *, forced_slot, method="B1", **options):
    arms = {arm: run_episode(episode, policy_factory(), intervention=arm,
                             forced_slot=forced_slot, method=method, **options) for arm in ("I0", "I1", "I2")}
    respond_equal = canonical(arms["I1"]["trace"]["observations"]["respond"]) == canonical(arms["I2"]["trace"]["observations"]["respond"])
    write_equal = canonical(arms["I0"]["trace"]["observations"]["write"]) == canonical(arms["I1"]["trace"]["observations"]["write"])
    if not respond_equal or not write_equal:
        raise RuntimeError("INTERVENTION_INFORMATION_BOUNDARY_FAILED")
    return {"arms": arms, "invariants": {"respond_I1_I2_identical": respond_equal, "write_I0_I1_identical": write_equal},
            "effects": {"Delta_actual": arms["I1"]["metrics"]["C2"] - arms["I0"]["metrics"]["C2"],
                        "Delta_persistencia": arms["I2"]["metrics"]["FJS"] - arms["I1"]["metrics"]["FJS"],
                        "Delta_total_futuro": arms["I2"]["metrics"]["FJS"] - arms["I0"]["metrics"]["FJS"],
                        "Delta_utilidad": arms["I2"]["rewards_by_component"]["R_long"] - arms["I0"]["rewards_by_component"]["R_long"]}}
