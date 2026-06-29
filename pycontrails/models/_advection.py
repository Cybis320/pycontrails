"""Shared Lagrangian advection helpers.

This private module holds the centerline integrator and the co-located
interpolation helper used by both :class:`pycontrails.models.dry_advection.DryAdvection`
and :class:`pycontrails.models.cocip.Cocip`. It is imported by both models and
imports neither, which avoids the ``dry_advection`` ↔ ``cocip`` import cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import geo, units


def interp_colocated(
    mdas: list[MetDataArray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    level: np.ndarray,
    time: np.ndarray,
    **interp_kwargs: Any,
) -> list[np.ndarray]:
    """Interpolate several co-located met variables, searching the grid once.

    All ``mdas`` share the met grid and are sampled at the same 4-D point, so the
    grid-index search (``_find_indices``, the dominant interpolation cost) is done
    once and reused for the remaining variables. This is bit-identical to
    interpolating each variable independently.

    Index reuse is unsupported with ``localize=True``; in that case each variable is
    interpolated independently. ``interp_kwargs`` must not contain ``q_method`` or
    ``use_indices`` (those belong to the high-level interpolation glue, not
    :meth:`MetDataArray.interpolate`).
    """
    share = not interp_kwargs.get("localize", False)
    out: list[np.ndarray] = []
    idx = None
    for mda in mdas:
        if not share:
            out.append(mda.interpolate(longitude, latitude, level, time, **interp_kwargs))
        elif idx is None:
            val, idx = mda.interpolate(
                longitude, latitude, level, time, return_indices=True, **interp_kwargs
            )
            out.append(val)
        else:
            out.append(
                mda.interpolate(longitude, latitude, level, time, indices=idx, **interp_kwargs)
            )
    return out


def advect_centerline_rk4(
    met: MetDataset,
    vector: GeoVectorDataset,
    t: np.datetime64,
    *,
    extra_dp_dt: npt.NDArray[np.floating] | float,
    **interp_kwargs: Any,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Integrate the plume centerline over one step with classical RK4.

    The centerline obeys the ordinary differential equation

    .. math::

        \frac{d\lambda}{dt} = \frac{u}{(N(\phi) + z) \cos\phi}, \quad
        \frac{d\phi}{dt} = \frac{v}{M(\phi) + z}, \quad
        \frac{dp}{dt} = \frac{\omega + \text{extra}}{100}

    where :math:`(u, v, \omega)` are re-interpolated from ``met`` at each of the four
    RK4 stages (start, two midpoints, end) and the intermediate positions. This
    replaces the previous forward-Euler step, which froze the wind at the start of
    the step and accrued kilometre-scale error over 1-3 h in curving flow.
    ``extra_dp_dt`` is the per-step-constant pressure tendency from sedimentation,
    [:math:`Pa \ s^{-1}`].

    Coordinates accumulate in ``float64`` to avoid a random-walk error over the many
    steps of a multi-hour integration.

    Parameters
    ----------
    met : MetDataset
        Meteorology providing ``eastward_wind``, ``northward_wind`` and
        ``lagrangian_tendency_of_air_pressure``.
    vector : GeoVectorDataset
        Waypoints at the start of the step. Start-of-step winds (``u_wind``,
        ``v_wind``, ``vertical_velocity``) are reused for the first RK4 stage.
    t : np.datetime64
        Target time at the end of the step.
    extra_dp_dt : npt.NDArray[np.floating] | float
        Constant pressure tendency added to the interpolated vertical velocity at
        every stage, [:math:`Pa \ s^{-1}`].
    **interp_kwargs : Any
        Interpolation keyword arguments forwarded to :meth:`intersect_met`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        New longitude, latitude and pressure level, [:math:`\deg`, :math:`\deg`,
        :math:`hPa`].
    """
    # ``q_method`` and ``use_indices`` belong to the high-level interpolation glue;
    # drop them so the remaining kwargs are accepted by ``MetDataArray.interpolate``,
    # which is called directly below to share grid indices (a fresh local dict).
    interp_kwargs.pop("q_method", None)
    interp_kwargs.pop("use_indices", None)

    # float64 accumulation (a float32 coordinate random-walks over hundreds of steps)
    lon0 = vector["longitude"].astype(np.float64)
    lat0 = vector["latitude"].astype(np.float64)
    lev0 = np.asarray(vector.level, dtype=np.float64)
    time0 = vector["time"]

    dt_td = t - time0
    dt_s = units.dt_to_seconds(dt_td)
    t_half = time0 + dt_td / 2
    t_full = np.full(lon0.shape, t)

    deg_per_rad = 180.0 / np.pi

    u_mda = met["eastward_wind"]
    v_mda = met["northward_wind"]
    w_mda = met["lagrangian_tendency_of_air_pressure"]

    def rates(
        lon: np.ndarray,
        lat: np.ndarray,
        lev: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        altitude = units.pl_to_m(lev)
        r_ew = geo.prime_vertical_radius_of_curvature(lat, altitude)
        r_ns = geo.meridional_radius_of_curvature(lat, altitude)
        cos_lat = np.cos(units.degrees_to_radians(lat))
        dlon = deg_per_rad * u / (r_ew * cos_lat)
        dlat = deg_per_rad * v / r_ns
        dlev = (w + extra_dp_dt) / 100.0
        return dlon, dlat, dlev

    def interp(
        lon: np.ndarray, lat: np.ndarray, lev: np.ndarray, time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u, v and omega are co-located; share the grid-index search across them.
        u, v, w = interp_colocated([u_mda, v_mda, w_mda], lon, lat, lev, time, **interp_kwargs)
        return u, v, w

    # Stage 1 reuses the once-per-step interpolation already attached to ``vector``.
    k1 = rates(lon0, lat0, lev0, vector["u_wind"], vector["v_wind"], vector["vertical_velocity"])

    half = dt_s / 2.0
    lon_a, lat_a, lev_a = lon0 + half * k1[0], lat0 + half * k1[1], lev0 + half * k1[2]
    k2 = rates(lon_a, lat_a, lev_a, *interp(lon_a, lat_a, lev_a, t_half))

    lon_b, lat_b, lev_b = lon0 + half * k2[0], lat0 + half * k2[1], lev0 + half * k2[2]
    k3 = rates(lon_b, lat_b, lev_b, *interp(lon_b, lat_b, lev_b, t_half))

    lon_c, lat_c, lev_c = lon0 + dt_s * k3[0], lat0 + dt_s * k3[1], lev0 + dt_s * k3[2]
    k4 = rates(lon_c, lat_c, lev_c, *interp(lon_c, lat_c, lev_c, t_full))

    sixth = dt_s / 6.0
    lon2 = lon0 + sixth * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    lat2 = lat0 + sixth * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    lev2 = lev0 + sixth * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])

    lon2 = (lon2 + 180.0) % 360.0 - 180.0  # wrap antimeridian
    return lon2, lat2, lev2
