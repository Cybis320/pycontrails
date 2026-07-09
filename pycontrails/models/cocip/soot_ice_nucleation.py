"""Ice-supersaturation (ISSR) contrail-onset pathway.

The *additive, opt-in* alternative onset: a persistent contrail may form wherever the
ambient air is ice-supersaturated (RHi > 1) and there is aircraft aerosol to nucleate on,
**even where the Schmidt-Appleman criterion (SAC) is not satisfied**. Enabled by
:attr:`CocipParams.soot_ice_nucleation`; with the flag off (default) none of this runs and
:class:`Cocip` is bit-identical to canonical CoCiP, so canonical results are preserved and
the ``"soot_issr"``-tagged waypoints are exactly the outliers relative to canonical SAC.

**On the ice-former (lit-informed reframe).** This pathway was conceived as soot
*deposition* nucleation, but recent laboratory measurements (Mahrt et al., 2024, *Atmos.
Chem. Phys.*) find aviation soot is a *poor* ice nucleus at cirrus temperatures: it
activates only at or above the homogeneous-freezing threshold of solution droplets
(RHi ~1.45), and only a small fraction of large aggregates below it. So the ice-former of
the observed SAC-false contrails is more likely ambient homogeneous freezing or
lubrication-oil droplets than soot itself -- all of which give an ice **number** close to
the familiar emissions/SAC number rather than the soot-poor extreme. Accordingly this
first-cut uses:

- an **onset** at ``RHi > RHI_ONSET_THRESHOLD`` (ISSR), agnostic to the ice-former, and
- an initial ice **number** from the emitted aerosol (:data:`ICE_ACTIVATION_FRACTION`),
  i.e. SAC-like, so downstream sedimentation is the validated surrogate (no soot
  correction by default; :func:`contrail_properties.emulated_crystal_radius`).

The two module constants are the tunables to calibrate against observations (raise the
threshold toward ~1.45 for the soot/homogeneous picture; lower the activation fraction for a
soot-poor / few-large-crystals sensitivity).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.physics import thermo

#: RHi (ratio, 1 = ice saturation) above which the ISSR onset forms a contrail. 1.0 forms
#: wherever the air is ice-supersaturated; the literature soot/homogeneous threshold is ~1.45.
RHI_ONSET_THRESHOLD = 1.0

#: Fraction of the emitted aerosol number that becomes ice crystals for the ISSR onset.
#: ~1 is SAC-like (the lit-informed default); lower it for a soot-poor sensitivity.
ICE_ACTIVATION_FRACTION = 1.0


def soot_issr_onset(
    specific_humidity: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    nvpm_ei_n: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    r"""ISSR onset criterion: ice-supersaturated air with aircraft aerosol present.

    Returns ``True`` at waypoints where the ambient is ice-supersaturated
    (:math:`\mathrm{RHi} = q / q_\mathrm{sat,ice}(T,p) > \mathrm{RHI\_ONSET\_THRESHOLD}`)
    and there is aerosol to nucleate on (``nvpm_ei_n > 0``), forming a contrail
    *independently of the SAC*. The caller combines this with the SAC mask (``sac | soot``)
    and tags the onset mechanism, so a ``True`` at a SAC-satisfied point is harmless (it is
    tagged ``"sac"``). See the module docstring for the ice-former reframe.

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
        Mask, ``True`` where the ISSR onset forms a contrail.
    """
    rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)
    return (rhi > RHI_ONSET_THRESHOLD) & (np.asarray(nvpm_ei_n, dtype=float) > 0.0)


def deposition_initial_ice(
    nvpm_ei_n: npt.NDArray[np.floating],
    fuel_dist: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    specific_humidity: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Compute initial ice number and ice water content for the ISSR-onset route.

    The initial condition for ISSR-onset contrails, distinct from the SAC droplet-freezing
    route (:func:`contrail_properties.initial_iwc`, which assumes exhaust water reaching
    liquid saturation):

    - ``n_ice_per_m`` = :data:`ICE_ACTIVATION_FRACTION` :math:`\cdot` ``nvpm_ei_n``
      :math:`\cdot` ``fuel_dist`` -- the emitted aerosol number (SAC-like by default; see the
      module docstring for why the soot-poor extreme is unlikely for observed contrails);
    - ``iwc`` = the **ambient** excess vapour over ice saturation,
      :math:`\max(q - q_\mathrm{sat,ice}(T, p), 0)`, deposited onto the nucleated crystals
      (rather than emitted exhaust water).

    Downstream growth (:func:`contrail_properties.new_ice_water_content`), sedimentation and
    the wake downwash are the existing machinery, shared with the SAC route.

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
        ``(n_ice_per_m, iwc)`` for the ISSR-onset waypoints.
    """
    q_sat_ice = thermo.q_sat_ice(air_temperature, air_pressure)
    iwc = np.maximum(np.asarray(specific_humidity, dtype=float) - q_sat_ice, 0.0)
    n_ice_per_m = ICE_ACTIVATION_FRACTION * np.asarray(nvpm_ei_n, dtype=float) * fuel_dist
    return n_ice_per_m, iwc
