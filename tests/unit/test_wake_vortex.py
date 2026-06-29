"""Tests for :mod:`pycontrails.models.cocip.wake_vortex`."""

from __future__ import annotations

import numpy as np

from pycontrails.models.cocip import wake_vortex
from pycontrails.physics import thermo

# Representative cruise conditions (~FL350) for a 2-waypoint "fleet".
_TAS = np.array([240.0, 240.0])
_T = np.array([219.0, 219.0])
_P = np.array([23842.0, 23842.0])
_DS_DZ = np.array([0.004, 0.004])
_KW = dict(
    effective_vertical_resolution=2000.0,
    wind_shear_enhancement_exponent=0.5,
    turbulent_vertical_velocity_scale=0.1,
)


def _profile(time, *, wingspan=60.0, mass=None, dT_dz=0.003, fraction=0.25):
    mass = np.array([300_000.0, 180_000.0]) if mass is None else mass
    return wake_vortex.downward_displacement_profile(
        np.asarray(time, dtype=float),
        wingspan,
        _TAS,
        mass,
        _T,
        np.full(2, dT_dz),
        _DS_DZ,
        _P,
        displacement_fraction=fraction,
        **_KW,
    )


def test_downward_displacement_profile_shape_and_endpoints() -> None:
    """Profile has shape (n_waypoints, n_times); starts at 0 and lands at fraction*dz_max."""
    t = np.array([0.0, 5.0, 30.0, 120.0, 1200.0])  # last time is well past t_max
    z = _profile(t)
    assert z.shape == (2, 5)

    # Starts at the formation altitude (zero displacement).
    np.testing.assert_allclose(z[:, 0], 0.0, atol=1e-9)

    # Lands at displacement_fraction * dz_max (here 0.25, Schumann Eq 13).
    dz_max = wake_vortex.max_downward_displacement(
        60.0, _TAS, np.array([300_000.0, 180_000.0]), _T, np.full(2, 0.003), _DS_DZ, _P, **_KW
    )
    np.testing.assert_allclose(z[:, -1], 0.25 * dz_max, rtol=1e-6)


def test_downward_displacement_profile_monotone_and_initial_slope() -> None:
    """Descent is monotone-decelerating with initial centroid velocity = fraction * w0."""
    t = np.linspace(0.0, 600.0, 1201)
    z = _profile(t)
    # Monotone non-decreasing (never rebounds upward).
    assert np.all(np.diff(z, axis=1) >= -1e-9)
    # Decelerating: the per-step increment shrinks over the descent.
    incr = np.diff(z[0])
    assert incr[0] > incr[20] > incr[-1]

    # Initial slope ≈ fraction * w0, with w0 = b0 / t0.
    rho = thermo.rho_d(_T, _P)
    t0 = wake_vortex.effective_time_scale(60.0, _TAS, np.array([300_000.0, 180_000.0]), rho)
    w0 = wake_vortex.wake_vortex_separation(60.0) / t0
    slope0 = (z[:, 1] - z[:, 0]) / (t[1] - t[0])
    np.testing.assert_allclose(slope0, 0.25 * w0, rtol=0.05)


def test_downward_displacement_profile_fraction_one_is_vortex_core() -> None:
    """displacement_fraction=1 recovers the full vortex-core max sinking at the endpoint."""
    dz_max = wake_vortex.max_downward_displacement(
        60.0, _TAS, np.array([300_000.0, 180_000.0]), _T, np.full(2, 0.003), _DS_DZ, _P, **_KW
    )
    z = _profile(np.array([1200.0]), fraction=1.0)
    np.testing.assert_allclose(z[:, 0], dz_max, rtol=1e-6)


def test_downward_displacement_profile_regime_timing() -> None:
    """Strongly stratified air reaches the landing depth sooner than weakly stratified."""
    t = np.linspace(0.0, 1200.0, 2401)
    weak = _profile(t, dT_dz=0.0005)[0]  # small dT/dz -> small N_bv -> weakly stratified
    strong = _profile(t, dT_dz=0.02)[0]  # large dT/dz -> large N_bv -> strongly stratified

    def time_to_settle(profile):
        target = 0.99 * profile[-1]
        return t[np.argmax(profile >= target)]

    assert time_to_settle(strong) < time_to_settle(weak)
