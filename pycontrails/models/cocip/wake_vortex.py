"""Wave-vortex downwash functions.

This module includes equations from the original CoCiP model
:cite:`schumannContrailCirrusPrediction2012`. An alternative set of equations based on
:cite:`unterstrasserPropertiesYoungContrails2016` is available in
:py:mod:`unterstrasser_wake_vortex`.

Unterstrasser Notes
-------------------

Improved estimation of the survival fraction of the contrail ice crystal number ``f_surv``
during the  wake-vortex phase. This is a parameterised model that is developed based on
outputs provided by large eddy simulations.

For comparison, CoCiP assumes that ``f_surv`` is equal to the change in the contrail ice water
content (by mass) before and after the wake vortex phase. However, for larger (smaller) ice
particles, their survival fraction by number could be smaller (larger) than their survival fraction
by mass. This is particularly important in the "soot-poor" scenario, for example, in cleaner
lean-burn engines where their soot emissions can be 3-4 orders of magnitude lower than conventional
RQL engines.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.models.cocip import wind_shear
from pycontrails.physics import constants, thermo


def max_downward_displacement(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    dT_dz: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float,
    turbulent_vertical_velocity_scale: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement after the wake vortex phase.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    dT_dz : npt.NDArray[np.floating]
        potential temperature gradient, [:math:`K m^{-1}`]
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`
    turbulent_vertical_velocity_scale : npt.NDArray[np.floating] | float
        Passed through to :func:`turbulent_kinetic_energy_dissipation_rate`, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]

    References
    ----------
    - :cite:`holzapfelProbabilisticTwoPhaseWake2003`
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    t_0 = effective_time_scale(wingspan, true_airspeed, aircraft_mass, rho_air)

    dz_max_strong = downward_displacement_strongly_stratified(
        wingspan, true_airspeed, aircraft_mass, rho_air, n_bv
    )

    is_weakly_stratified = n_bv * t_0 < 0.8
    if isinstance(wingspan, np.ndarray):
        wingspan = wingspan[is_weakly_stratified]
    if isinstance(aircraft_mass, np.ndarray):
        aircraft_mass = aircraft_mass[is_weakly_stratified]

    dz_max_weak = downward_displacement_weakly_stratified(
        wingspan=wingspan,
        true_airspeed=true_airspeed[is_weakly_stratified],
        aircraft_mass=aircraft_mass,
        rho_air=rho_air[is_weakly_stratified],
        n_bv=n_bv[is_weakly_stratified],
        dz_max_strong=dz_max_strong[is_weakly_stratified],
        ds_dz=ds_dz[is_weakly_stratified],
        t_0=t_0[is_weakly_stratified],
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
        turbulent_vertical_velocity_scale=turbulent_vertical_velocity_scale,
    )

    dz_max_strong[is_weakly_stratified] = dz_max_weak
    return dz_max_strong


def effective_time_scale(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective time scale of the wake vortex.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m \ s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg \ m^{-3}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Wake vortex effective time scale, [:math:`s`]

    Notes
    -----
    See section 2.5 (pg 547) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = np.pi**4 / 32
    return c * wingspan**3 * rho_air * true_airspeed / (aircraft_mass * constants.g)


def downward_displacement_strongly_stratified(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
    n_bv: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement under strongly stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.floating]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Maximum contrail downward displacement, strongly stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 547 - 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (1.49 * 16) / (2 * np.pi**3)  # This is W2 in Schumann's Fortran code
    return (c * aircraft_mass * constants.g) / (wingspan**2 * rho_air * true_airspeed * n_bv)


def downward_displacement_weakly_stratified(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
    n_bv: npt.NDArray[np.floating],
    dz_max_strong: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    t_0: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float,
    turbulent_vertical_velocity_scale: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement under weakly/stably stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.floating]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]
    dz_max_strong : npt.NDArray[np.floating]
        Max contrail downward displacement under strongly stratified conditions, [:math:`m`]
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    t_0 : npt.NDArray[np.floating]
        Wake vortex effective time scale, [:math:`s`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`
    turbulent_vertical_velocity_scale : npt.NDArray[np.floating] | float
        Passed through to :func:`turbulent_kinetic_energy_dissipation_rate`, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Maximum contrail downward displacement, weakly/stably stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    b_0 = wake_vortex_separation(wingspan)
    dz_max = np.maximum(dz_max_strong, 10.0)
    shear_enhancement_factor = wind_shear.wind_shear_enhancement_factor(
        dz_max, effective_vertical_resolution, wind_shear_enhancement_exponent
    )

    # Calculate epsilon and epsilon star
    # In Schumann's Fortran code, epsn = EDR and epsn_st = EPSN
    epsn = turbulent_kinetic_energy_dissipation_rate(
        ds_dz, shear_enhancement_factor, turbulent_vertical_velocity_scale
    )
    epsn_st = normalized_dissipation_rate(epsn, wingspan, true_airspeed, aircraft_mass, rho_air)
    return b_0 * (7.68 * (1 - 4.07 * epsn_st + 5.67 * epsn_st**2) * (0.79 - n_bv * t_0) + 1.88)


def wake_vortex_separation(
    wingspan: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating] | float:
    """
    Calculate the wake vortex separation.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating] | float
        wake vortex separation, [:math:`m`]
    """
    return (np.pi * wingspan) / 4.0


def turbulent_kinetic_energy_dissipation_rate(
    ds_dz: npt.NDArray[np.floating],
    shear_enhancement_factor: npt.NDArray[np.floating] | float = 1.0,
    turbulent_vertical_velocity_scale: npt.NDArray[np.floating] | float = 0.1,
) -> npt.NDArray[np.floating]:
    """
    Calculate the turbulent kinetic energy dissipation rate (epsilon).

    The shear enhancement factor is used to account for any sub-grid scale turbulence.

    Parameters
    ----------
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    shear_enhancement_factor : npt.NDArray[np.floating] | float
        Multiplication factor to enhance the wind shear
    turbulent_vertical_velocity_scale : npt.NDArray[np.floating] | float
        Turbulent vertical velocity scale, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]

    Notes
    -----
    - See eq. (37) in :cite:`schumannContrailCirrusPrediction2012`.
    - In a personal correspondence, Dr. Schumann identified a print error in Eq. (37)
      of the 2012 paper where the shear term should not be squared.
      The correct equation is listed in Eq. (13) :cite:`schumannTurbulentMixingStably1995`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`schumannTurbulentMixingStably1995`
    """
    return 0.5 * turbulent_vertical_velocity_scale**2 * (ds_dz * shear_enhancement_factor**2)


def normalized_dissipation_rate(
    epsilon: npt.NDArray[np.floating],
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the normalized dissipation rate of the sinking wake vortex.

    Parameters
    ----------
    epsilon: npt.NDArray[np.floating]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating] | float
        density of air for each waypoint, [:math:`kg m^{-3}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Normalized dissipation rate of the sinking wake vortex

    Notes
    -----
    See page 548 of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (np.pi / 4) ** (1 / 3) * np.pi**3 / 8  # This is W6 in Schumann's Fortran code
    numer = c * (epsilon * wingspan) ** (1 / 3) * wingspan**2 * rho_air * true_airspeed

    # epsn_st = epsilon star
    epsn_st = numer / (constants.g * aircraft_mass)

    # In a personal correspondence, Schumann gives the precise value
    # of 0.358906526 here
    # In the 2012 paper, Schumann gives 0.36
    # The precise value is likely insignificant because we don't expect epsn_st
    # to be larger than 0.36
    return np.minimum(epsn_st, 0.36)


def initial_contrail_width(
    wingspan: npt.NDArray[np.floating] | float, dz_max: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate the initial contrail width.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    dz_max : npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
        Only the size of this array is used; the values are ignored.

    Returns
    -------
    npt.NDArray[np.floating]
        Initial contrail width, [:math:`m`]
    """
    return np.full_like(dz_max, np.pi / 4) * wingspan


def initial_contrail_depth(
    dz_max: npt.NDArray[np.floating], initial_wake_vortex_depth: float | npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate the initial contrail depth.

    Parameters
    ----------
    dz_max : npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
    initial_wake_vortex_depth : float | npt.NDArray[np.floating]
        Initial wake vortex depth scaling factor.
        Denoted `C_D0` in eq (14) in :cite:`schumannContrailCirrusPrediction2012`.

    Returns
    -------
    npt.NDArray[np.floating]
        Initial contrail depth, [:math:`m`]
    """
    return dz_max * initial_wake_vortex_depth


def downward_displacement_params(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    dT_dz: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float,
    turbulent_vertical_velocity_scale: npt.NDArray[np.floating] | float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Per-waypoint parameters of the wake-descent profile: ``(dz_max, t_max, p)``.

    The time-independent part of :func:`downward_displacement_profile`, split out so the
    descent can be evaluated cheaply at many times (e.g. once per integration step, to
    time-resolve the downwash) without recomputing the anchors. See that function for the
    physics and references.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]
        ``dz_max`` [:math:`m`], ``t_max`` [:math:`s`] (time of maximum sinking) and the
        decelerating shape exponent ``p``, each of shape ``(n_waypoints,)``.
    """
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    t_0 = effective_time_scale(wingspan, true_airspeed, aircraft_mass, rho_air)
    b_0 = wake_vortex_separation(wingspan)
    w_0 = b_0 / t_0  # w_0 = Γ_0 / (2π b_0) = b_0 / t_0
    dz_max = max_downward_displacement(
        wingspan,
        true_airspeed,
        aircraft_mass,
        air_temperature,
        dT_dz,
        ds_dz,
        air_pressure,
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
        turbulent_vertical_velocity_scale=turbulent_vertical_velocity_scale,
    )
    # Per-waypoint anchors as 1-D arrays (n_waypoints,).
    dz_max = np.atleast_1d(np.asarray(dz_max, dtype=float))
    t_0 = np.broadcast_to(np.atleast_1d(np.asarray(t_0, dtype=float)), dz_max.shape)
    w_0 = np.broadcast_to(np.atleast_1d(np.asarray(w_0, dtype=float)), dz_max.shape)
    n_bv = np.broadcast_to(np.atleast_1d(np.asarray(n_bv, dtype=float)), dz_max.shape)
    # Time of maximum sinking: 12*t_0 (weakly) -> 5*t_0 (strongly stratified).
    regime = np.clip(n_bv * t_0 / 0.8, 0.0, 1.0)  # 0 weak, 1 strong
    t_max = t_0 * (12.0 - 7.0 * regime)
    # Decelerating shape exponent; clip to >= 1 to keep the descent monotone-decelerating.
    p = np.maximum(w_0 * t_max / dz_max, 1.0)
    return dz_max, t_max, p


def downward_displacement_at(
    time_since_formation: npt.NDArray[np.floating],
    dz_max: npt.NDArray[np.floating],
    t_max: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating],
    *,
    displacement_fraction: float = 0.25,
) -> npt.NDArray[np.floating]:
    r"""Contrail-centroid downward displacement at given times from precomputed params.

    Evaluates the decelerating descent
    :math:`\mathrm{fraction} \cdot \Delta z_w [1 - (1 - t/t_\mathrm{max})^p]`, held at
    :math:`\Delta z_w` for :math:`t \geq t_\mathrm{max}`. ``time_since_formation`` and the
    params broadcast together (numpy rules), so this serves both the 2-D profile
    (:func:`downward_displacement_profile`) and an elementwise per-waypoint evaluation (as
    used to time-resolve the downwash a step at a time).

    Parameters
    ----------
    time_since_formation : npt.NDArray[np.floating]
        Age since formation, [:math:`s`].
    dz_max, t_max, p : npt.NDArray[np.floating]
        Profile parameters from :func:`downward_displacement_params`.
    displacement_fraction : float
        Centroid fraction of the vortex-core sinking. Defaults to ``0.25``.

    Returns
    -------
    npt.NDArray[np.floating]
        Downward displacement of the contrail centroid, [:math:`m`]. Positive is downward.
    """
    t = np.asarray(time_since_formation, dtype=float)
    tau = np.minimum(t, t_max) / t_max
    z_vortex = dz_max * (1.0 - (1.0 - tau) ** p)
    return displacement_fraction * z_vortex


def downward_displacement_profile(
    time_since_formation: npt.NDArray[np.floating],
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    dT_dz: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float,
    turbulent_vertical_velocity_scale: npt.NDArray[np.floating] | float,
    *,
    displacement_fraction: float = 0.25,
) -> npt.NDArray[np.floating]:
    r"""Time-resolved downward displacement of the contrail centroid during the wake phase.

    :func:`max_downward_displacement` returns only the end-state maximum sinking
    :math:`\Delta z_w`; :cite:`schumannContrailCirrusPrediction2012` explicitly "do[es]
    not resolve the details of the jet and wake dynamics in the first minutes". This
    function resolves the centroid descent over time so the early, fast-moving wake
    phase can be sampled (e.g. at 5 s) for geometry-grade traces.

    The descent is built from the same anchors CoCiP already uses:

    - initial velocity scale :math:`w_0 = b_0 / t_0` (with :math:`b_0` the vortex
      separation and :math:`t_0` the :func:`effective_time_scale`),
    - maximum sinking :math:`\Delta z_w` (:func:`max_downward_displacement`),
    - the time of maximum sinking, which :cite:`schumannContrailCirrusPrediction2012`
      (p. 548, after :cite:`holzapfelProbabilisticTwoPhaseWake2003`) places at
      :math:`5 t_0` (strongly stratified) to :math:`12 t_0` (weakly stratified). This is
      interpolated on the regime parameter :math:`N_{BV} t_0` (the same 0.8 threshold
      used for :math:`\Delta z_w`).

    The vortex-core descent is the decelerating curve

    .. math::

        z_v(t) = \Delta z_w \left[1 - (1 - t/t_\mathrm{max})^p\right], \quad
        p = \frac{w_0 \, t_\mathrm{max}}{\Delta z_w},

    which honours :math:`\dot z_v(0) = w_0`, :math:`z_v(t_\mathrm{max}) = \Delta z_w`, and
    :math:`\dot z_v(t_\mathrm{max}) = 0` (smooth landing); ``p`` is clipped to
    :math:`\geq 1` so the descent is always monotone-decelerating. The contrail
    **centroid** is placed at ``displacement_fraction`` of the vortex-core sinking
    (:math:`C_{z1} = 0.25`, Eq. 13 of :cite:`schumannContrailCirrusPrediction2012` — the
    value CoCiP uses for the post-wake contrail altitude).

    .. note::

        The endpoints (:math:`w_0`, :math:`\Delta z_w`, :math:`t_\mathrm{max}`) and the
        :math:`0.25` centroid fraction are from the literature. The interpolating *shape*
        between them is a closure: the wake phase is intrinsically probabilistic and the
        cited models give the end-state, not the trajectory. Treat the transient shape as
        model-dependent (validatable only at its endpoints without observations).

    Parameters
    ----------
    time_since_formation : npt.NDArray[np.floating]
        Times since contrail formation at which to evaluate the descent, [:math:`s`].
        1-D array of length ``n_times``.
    wingspan, true_airspeed, aircraft_mass, air_temperature, dT_dz, ds_dz, air_pressure :
        Per-waypoint quantities, exactly as passed to :func:`max_downward_displacement`.
        Array-valued quantities have length ``n_waypoints``.
    effective_vertical_resolution : float
        Passed through to :func:`max_downward_displacement`.
    wind_shear_enhancement_exponent, turbulent_vertical_velocity_scale :
        Passed through to :func:`max_downward_displacement`.
    displacement_fraction : float
        Fraction of the vortex-core sinking at which the contrail centroid settles.
        Defaults to ``0.25`` (:math:`C_{z1}`, Eq. 13).

    Returns
    -------
    npt.NDArray[np.floating]
        Downward displacement of the contrail centroid, shape
        ``(n_waypoints, n_times)``, [:math:`m`]. Positive is downward.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`holzapfelProbabilisticTwoPhaseWake2003`
    """
    dz_max, t_max, p = downward_displacement_params(
        wingspan,
        true_airspeed,
        aircraft_mass,
        air_temperature,
        dT_dz,
        ds_dz,
        air_pressure,
        effective_vertical_resolution,
        wind_shear_enhancement_exponent,
        turbulent_vertical_velocity_scale,
    )
    t = np.atleast_1d(np.asarray(time_since_formation, dtype=float))  # (n_times,)
    return downward_displacement_at(
        t[None, :],
        dz_max[:, None],
        t_max[:, None],
        p[:, None],
        displacement_fraction=displacement_fraction,
    )
