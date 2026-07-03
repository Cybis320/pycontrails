"""Soot-in-ISSR ice-nucleation onset pathway (**research stubs**).

This module holds the *additive, opt-in* alternative contrail-onset pathway: soot
particles acting as depositional/immersion ice nuclei in ice-supersaturated air
(RHi > 100%), forming persistent contrails even where the Schmidt-Appleman criterion
(SAC) is not satisfied.

It is enabled by :attr:`CocipParams.soot_ice_nucleation`. With that flag off (the
default) none of this module runs and :class:`Cocip` is bit-identical to canonical
CoCiP — canonical results are preserved, and the ``"soot_issr"``-tagged waypoints from
an enabled run are exactly the outliers relative to canonical SAC.

.. warning::

    The two functions below are **unimplemented research hooks**. They encode model
    choices (the nucleation criterion, the soot ice-nucleating fraction, and the
    deposition-nucleation initial condition) that must be specified and validated
    against observations before use. They raise :class:`NotImplementedError` so the
    pathway cannot silently produce unvalidated results.

Distinction from the existing droplet route: :mod:`pycontrails.models.extended_k15`
(and :func:`contrail_properties.ice_particle_activation_rate`) model the *Koehler
droplet-activation* pathway (the SAC liquid route). The pathway here is *deposition /
immersion nucleation directly on soot* in ISSR — a different microphysical onset.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def soot_issr_onset(
    specific_humidity: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    nvpm_ei_n: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    r"""Determine where soot nucleates ice in ice-supersaturated air (onset criterion).

    **Research hook — unimplemented.** Encodes the first two of the three knobs:
    the onset criterion and the soot ice-nucleating fraction.

    The implementation should return ``True`` at waypoints where soot acts as an
    effective depositional/immersion ice nucleus, forming a contrail *independently of
    the SAC*. The caller combines this with the SAC mask (``sac | soot``) and tags the
    onset mechanism, so returning ``True`` at SAC-satisfied points is harmless (those
    are tagged ``"sac"``).

    A physically-motivated implementation is expected to require, at minimum:

    - ice supersaturation, :math:`\mathrm{RHi} = q / q_\mathrm{sat,ice}(T, p) > 1`
      (compute via :func:`pycontrails.physics.thermo.rhi`);
    - a non-negligible soot number (``nvpm_ei_n`` above a threshold);
    - possibly a temperature dependence of soot deposition-nucleation efficiency, and
      an ice-nucleating *fraction* of the soot population as a function of RHi and T.

    Parameters
    ----------
    specific_humidity : npt.NDArray[np.floating]
        Ambient specific humidity, [:math:`kg \ kg^{-1}`].
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature, [:math:`K`].
    air_pressure : npt.NDArray[np.floating]
        Ambient pressure, [:math:`Pa`].
    nvpm_ei_n : npt.NDArray[np.floating]
        Non-volatile PM (soot) number emission index, [:math:`kg^{-1}`].

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask, ``True`` where soot nucleates ice in ISSR.
    """
    raise NotImplementedError(
        "soot_issr_onset is a research hook: implement the soot deposition-nucleation "
        "onset criterion (RHi>1 + soot ice-nucleating fraction) and validate against "
        "observations before enabling CocipParams.soot_ice_nucleation."
    )


def deposition_initial_ice(
    nvpm_ei_n: npt.NDArray[np.floating],
    fuel_dist: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    specific_humidity: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Compute initial ice number and ice water content for the soot-nucleation route.

    **Research hook — unimplemented.** The third knob: the initial condition for
    soot-nucleated contrails, which differs from the SAC droplet-freezing route
    (:func:`contrail_properties.initial_iwc`, which assumes exhaust water reaching
    liquid saturation).

    The implementation should return, for soot-onset waypoints:

    - ``n_ice_per_m``: the initial number of ice crystals per contrail metre, from the
      *ice-nucleating fraction* of the emitted soot
      (:math:`\approx f_\mathrm{IN} \cdot \mathrm{nvpm\_ei\_n} \cdot \mathrm{fuel\_dist}`);
    - ``iwc``: the initial ice water content from **ambient depositional growth** — the
      excess vapour over ice saturation, :math:`\max(q - q_\mathrm{sat,ice}(T, p), 0)`,
      deposited onto the nucleated crystals — rather than from emitted exhaust water.

    Downstream growth (:func:`contrail_properties.new_ice_water_content`), sedimentation
    (:func:`contrail_properties.ice_particle_terminal_fall_speed`) and the wake downwash
    are the existing rigorous machinery, shared with the SAC route.

    Parameters
    ----------
    nvpm_ei_n : npt.NDArray[np.floating]
        Soot number emission index, [:math:`kg^{-1}`].
    fuel_dist : npt.NDArray[np.floating]
        Fuel consumption per unit distance, [:math:`kg \ m^{-1}`].
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature, [:math:`K`].
    specific_humidity : npt.NDArray[np.floating]
        Ambient specific humidity, [:math:`kg \ kg^{-1}`].
    air_pressure : npt.NDArray[np.floating]
        Ambient pressure, [:math:`Pa`].

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        ``(n_ice_per_m, iwc)`` for the soot-onset waypoints.
    """
    raise NotImplementedError(
        "deposition_initial_ice is a research hook: implement the soot-nucleated initial "
        "ice number (from the ice-nucleating soot fraction) and initial iwc (from ambient "
        "deposition of the ISSR excess vapour) before enabling soot_ice_nucleation."
    )
