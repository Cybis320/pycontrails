"""Unit conversion support."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.physics import constants
from pycontrails.utils.types import ArrayScalarLike, support_arraylike


def pl_to_ft(pl: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert from pressure level (hPa) to altitude (ft).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    pl : ArrayScalarLike
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Returns
    -------
    ArrayScalarLike
        altitude, [:math:`ft`]

    See Also
    --------
    pl_to_m
    ft_to_pl
    m_to_T_isa
    """
    return m_to_ft(pl_to_m(pl))


def ft_to_pl(h: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert from altitude (ft) to pressure level (hPa).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    h : ArrayScalarLike
        altitude, [:math:`ft`]

    Returns
    -------
    ArrayScalarLike
        pressure level, [:math:`hPa`], [:math:`mbar`]

    See Also
    --------
    m_to_pl
    pl_to_ft
    m_to_T_isa
    """
    return m_to_pl(ft_to_m(h))


def kelvin_to_celsius(kelvin: ArrayScalarLike) -> ArrayScalarLike:
    """Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    kelvin : ArrayScalarLike
        temperature [:math:`K`]

    Returns
    -------
    ArrayScalarLike
        temperature [:math:`C`]
    """
    return kelvin + constants.absolute_zero


def m_to_T_isa(h: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the ambient temperature (K) for a given altitude (m).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    h : ArrayScalarLike
        altitude, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        ICAO standard atmosphere ambient temperature, [:math:`K`]


    References
    ----------
    - :cite:`wikipediacontributorsInternationalStandardAtmosphere2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/International_Standard_Atmosphere

    See Also
    --------
    m_to_pl
    ft_to_pl
    """
    h_min = np.minimum(h, constants.h_tropopause)
    return constants.T_msl + h_min * constants.T_lapse_rate  # type: ignore[return-value]


def _low_altitude_m_to_pl(h: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    T_isa = m_to_T_isa(h)
    power_term = -constants.g / (constants.T_lapse_rate * constants.R_d)
    return (constants.p_surface * (T_isa / constants.T_msl) ** power_term) / 100.0


def _high_altitude_m_to_pl(h: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    T_tropopause_isa = m_to_T_isa(np.asarray(constants.h_tropopause))
    power_term = -constants.g / (constants.T_lapse_rate * constants.R_d)
    p_tropopause_isa = constants.p_surface * (T_tropopause_isa / constants.T_msl) ** power_term
    inside_exp = (-constants.g / (constants.R_d * T_tropopause_isa)) * (h - constants.h_tropopause)
    return p_tropopause_isa * np.exp(inside_exp) / 100.0


@support_arraylike
def m_to_pl(h: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Convert from altitude (m) to pressure level (hPa).

    Parameters
    ----------
    h : npt.NDArray[np.floating]
        altitude, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
        pressure level, [:math:`hPa`], [:math:`mbar`]

    References
    ----------
    - :cite:`wikipediacontributorsBarometricFormula2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/Barometric_formula

    See Also
    --------
    m_to_T_isa
    ft_to_pl
    """
    condlist = [h < constants.h_tropopause, h >= constants.h_tropopause]
    funclist = [_low_altitude_m_to_pl, _high_altitude_m_to_pl, np.nan]  # nan passed through
    return np.piecewise(h, condlist, funclist)  # type: ignore[call-overload]


def _low_altitude_pl_to_m(pl: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    base = 100.0 * pl / constants.p_surface
    exponent = -constants.T_lapse_rate * constants.R_d / constants.g
    T_isa = constants.T_msl * base**exponent
    return (T_isa - constants.T_msl) / constants.T_lapse_rate


def _high_altitude_pl_to_m(pl: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    T_tropopause_isa = m_to_T_isa(np.asarray(constants.h_tropopause))
    power_term = -constants.g / (constants.T_lapse_rate * constants.R_d)
    p_tropopause_isa = constants.p_surface * (T_tropopause_isa / constants.T_msl) ** power_term
    inside_exp = np.log(pl * 100.0 / p_tropopause_isa)
    return inside_exp / (-constants.g / (constants.R_d * T_tropopause_isa)) + constants.h_tropopause


@support_arraylike
def pl_to_m(pl: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Convert from pressure level (hPa) to altitude (m).

    Function is slightly different from the classical formula:
    ``constants.T_msl / 0.0065) * (1 - (pl_pa / constants.p_surface) ** (1 / 5.255)``
    in order to provide a mathematical inverse to :func:`m_to_pl`.

    For low altitudes (below the tropopause), this implementation closely agrees to classical
    formula.

    Parameters
    ----------
    pl : ArrayLike
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Returns
    -------
    ArrayLike
        altitude, [:math:`m`]

    References
    ----------
    - :cite:`wikipediacontributorsBarometricFormula2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/Barometric_formula

    See Also
    --------
    pl_to_ft
    m_to_pl
    m_to_T_isa
    """
    pl_tropopause = m_to_pl(constants.h_tropopause)
    condlist = [pl < pl_tropopause, pl >= pl_tropopause]
    funclist = [_high_altitude_pl_to_m, _low_altitude_pl_to_m, np.nan]  # nan passed through
    return np.piecewise(pl, condlist, funclist)  # type: ignore[call-overload]


def degrees_to_radians(degrees: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert from degrees to radians.

    Parameters
    ----------
    degrees : ArrayScalarLike
        Degrees values, [:math:`\deg`]

    Returns
    -------
    ArrayScalarLike
        Radians values
    """
    return degrees * (np.pi / 180.0)


def radians_to_degrees(radians: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert from radians to degrees.

    Parameters
    ----------
    radians : ArrayScalarLike
        degrees values, [:math:`\rad`]

    Returns
    -------
    ArrayScalarLike
        Radian values
    """
    return radians * (180.0 / np.pi)


@support_arraylike
def normal_gravity(latitude: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Calculate WGS-84 normal (sea-level) gravity at a given latitude.

    Uses the closed-form Somigliana expression for normal gravity on the surface
    of the WGS-84 reference ellipsoid

    .. math::

        g(\phi) = g_e \frac{1 + k \sin^2\phi}{\sqrt{1 - e^2 \sin^2\phi}}

    This is the *true* latitude-varying acceleration of gravity. It differs from
    the fixed WMO standard value :attr:`constants.g` (:math:`9.80665 \ m \ s^{-2}`,
    the approximate value at :math:`45 \deg` latitude) used elsewhere in this
    module for pressure-altitude conversions.

    Parameters
    ----------
    latitude : npt.NDArray[np.floating]
        Geodetic latitude, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.floating]
        Normal gravity at mean sea level, [:math:`m \ s^{-2}`]

    Notes
    -----
    Coefficients are the WGS-84 defining constants: equatorial normal gravity
    :math:`g_e = 9.7803253359 \ m \ s^{-2}`, the Somigliana constant
    :math:`k = 0.00193185265241`, and the squared first eccentricity
    :math:`e^2 = 0.00669437999014`. See
    https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation.

    See Also
    --------
    geopotential_to_geometric_height
    """
    # WGS-84 Somigliana normal gravity coefficients
    g_e = 9.7803253359  # equatorial normal gravity, [m s-2]
    k = 0.00193185265241  # Somigliana formula constant
    e2 = 0.00669437999014  # square of the first eccentricity of the ellipsoid

    sin2 = np.sin(degrees_to_radians(latitude)) ** 2
    return g_e * (1.0 + k * sin2) / np.sqrt(1.0 - e2 * sin2)


def _effective_earth_radius(latitude: ArrayScalarLike, gravity: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate the latitude-dependent effective Earth radius.

    List (1968) / Mahoney effective radius used to convert a geopotential height
    into a geometric height by accounting for the curvature of the Earth

    .. math::

        R(\phi) = \frac{2 \, g(\phi)}
        {3.085462 \times 10^{-6} + 2.27 \times 10^{-9} \cos 2\phi
        - 2.0 \times 10^{-12} \cos 4\phi}

    Parameters
    ----------
    latitude : ArrayScalarLike
        Geodetic latitude, [:math:`\deg`]
    gravity : ArrayScalarLike
        WGS-84 normal gravity :math:`g(\phi)` at ``latitude`` (from
        :func:`normal_gravity`). Passed in by the caller so it is not recomputed.

    Returns
    -------
    ArrayScalarLike
        Effective Earth radius, [:math:`m`]. ``R(45) ~ 6.356e6``.
    """
    lat_rad = degrees_to_radians(latitude)
    denominator = 3.085462e-6 + 2.27e-9 * np.cos(2.0 * lat_rad) - 2.0e-12 * np.cos(4.0 * lat_rad)
    return 2.0 * gravity / denominator


def geopotential_to_geometric_height(
    geopotential: ArrayScalarLike, latitude: ArrayScalarLike
) -> ArrayScalarLike:
    r"""Convert geopotential to geometric height above the geoid.

    The ERA5 ``z`` field is a geopotential :math:`\Phi` (:math:`m^2 \ s^{-2}`),
    i.e. :math:`\int g \, \mathrm{d}z` with the true latitude-varying gravity
    already integrated by the model. The standard WMO geopotential-height
    diagnostic recovers a height as :math:`\Phi / g_0` using the fixed constant
    :math:`g_0 = 9.80665 \ m \ s^{-2}` (:attr:`constants.g`). That is a correct
    geopotential-height diagnostic but **not** a true geometric altitude: a single
    :math:`45 \deg` gravity value mis-scales the height by :math:`g_0 / g(\phi)`,
    a latitude-systematic error that flips sign about :math:`45 \deg` (about
    :math:`+30 \ m` at the equator, :math:`0` near :math:`45 \deg`, and
    :math:`-29 \ m` at the pole at :math:`11 \ km`).

    This converter therefore intentionally divides by the latitude-dependent
    normal gravity :func:`normal_gravity` rather than by the fixed
    :attr:`constants.g`, then applies a curvature correction using the effective
    Earth radius :func:`_effective_earth_radius`

    .. math::

        H = \frac{\Phi}{g(\phi)}, \qquad
        z_{\mathrm{geom}} = H \frac{R(\phi)}{R(\phi) - H}

    Parameters
    ----------
    geopotential : ArrayScalarLike
        Geopotential :math:`\Phi` (the ERA5 ``z`` field), [:math:`m^2 \ s^{-2}`]
    latitude : ArrayScalarLike
        Geodetic latitude, broadcast against ``geopotential``, [:math:`\deg`]

    Returns
    -------
    ArrayScalarLike
        Geometric height above the geoid (orthometric / MSL height), [:math:`m`].
        No geoid undulation is added and no conversion to WGS-84 ellipsoidal
        height is performed; the caller owns the geoid datum.

    See Also
    --------
    normal_gravity
    """
    gravity = normal_gravity(latitude)
    geopotential_height = geopotential / gravity
    r_eff = _effective_earth_radius(latitude, gravity)
    return geopotential_height * r_eff / (r_eff - geopotential_height)


def ft_to_m(ft: ArrayScalarLike) -> ArrayScalarLike:
    """Convert length from feet to meter.

    Parameters
    ----------
    ft : ArrayScalarLike
        length, [:math:`ft`]

    Returns
    -------
    ArrayScalarLike
        length, [:math:`m`]
    """
    return ft * 0.3048


def m_to_ft(m: ArrayScalarLike) -> ArrayScalarLike:
    """Convert length from meters to feet.

    Parameters
    ----------
    m : ArrayScalarLike
        length, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        length, [:math:`ft`]
    """
    return m / 0.3048


def m_per_s_to_knots(m_per_s: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert speed from meters per second (m/s) to knots.

    Parameters
    ----------
    m_per_s : ArrayScalarLike
        Speed, [:math:`m \ s^{-1}`]

    Returns
    -------
    ArrayScalarLike
        Speed, [:math:`knots`]
    """
    return m_per_s / 0.514444


def knots_to_m_per_s(knots: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert speed from knots to meters per second (m/s).

    Parameters
    ----------
    knots : ArrayScalarLike
        Speed, [:math:`knots`]

    Returns
    -------
    ArrayScalarLike
        Speed, [:math:`m \ s^{-1}`]
    """
    return knots * 0.514444


def longitude_distance_to_m(
    distance_degrees: ArrayScalarLike, latitude_mean: ArrayScalarLike
) -> ArrayScalarLike:
    r"""
    Convert longitude degrees distance between two points to cartesian distances in meters.

    Parameters
    ----------
    distance_degrees : ArrayScalarLike
        longitude distance, [:math:`\deg`]
    latitude_mean : ArrayScalarLike, optional
        mean latitude between ``longitude_1`` and ``longitude_2``, [:math:`\deg`]

    Returns
    -------
    ArrayScalarLike
        cartesian distance along the longitude axis, [:math:`m`]
    """
    latitude_mean_rad = degrees_to_radians(latitude_mean)
    return (distance_degrees / 180.0) * np.pi * constants.radius_earth * np.cos(latitude_mean_rad)


def latitude_distance_to_m(distance_degrees: ArrayScalarLike) -> ArrayScalarLike:
    r"""
    Convert latitude degrees distance between two points to cartesian distances in meters.

    Parameters
    ----------
    distance_degrees : ArrayScalarLike
        latitude distance, [:math:`\deg`]

    Returns
    -------
    ArrayScalarLike
        Cartesian distance along the latitude axis, [:math:`m`]
    """
    return (distance_degrees / 180.0) * np.pi * constants.radius_earth


def m_to_longitude_distance(
    distance_m: ArrayScalarLike, latitude_mean: ArrayScalarLike
) -> ArrayScalarLike:
    r"""
    Convert cartesian distance (meters) to differences in longitude degrees.

    Small angle approximation for ``distance_m`` << :attr:`constants.radius_earth`

    Parameters
    ----------
    distance_m : ArrayScalarLike
        cartesian distance along longitude axis, [:math:`m`]
    latitude_mean : ArrayScalarLike
        mean latitude between ``longitude_1`` and ``longitude_2``, [:math:`\deg`]

    Returns
    -------
    ArrayScalarLike
        longitude distance, [:math:`\deg`]
    """
    return radians_to_degrees(
        distance_m / (constants.radius_earth * np.cos(degrees_to_radians(latitude_mean)))
    )


def m_to_latitude_distance(distance_m: ArrayScalarLike) -> ArrayScalarLike:
    r"""
    Convert cartesian distance (meters) to differences in latitude degrees.

    Small angle approximation for ``distance_m`` << :attr:`constants.radius_earth`

    Parameters
    ----------
    distance_m : ArrayScalarLike
        cartesian distance along latitude axis, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        latitude distance, [:math:`\deg`]
    """
    return radians_to_degrees(distance_m / constants.radius_earth)


def tas_to_mach_number(true_airspeed: ArrayScalarLike, T: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate Mach number from true airspeed at a specified ambient temperature.

    Parameters
    ----------
    true_airspeed : ArrayScalarLike
        True airspeed, [:math:`m \ s^{-1}`]
    T : ArrayScalarLike
        Ambient temperature, [:math:`K`]

    Returns
    -------
    ArrayScalarLike
        Mach number, [:math: `Ma`]

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    return true_airspeed / np.sqrt((constants.kappa * constants.R_d) * T)


def mach_number_to_tas(
    mach_number: float | npt.NDArray[np.floating], T: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    r"""Calculate true airspeed from the Mach number at a specified ambient temperature.

    Parameters
    ----------
    mach_number : float | npt.NDArray[np.floating]
        Mach number, [:math: `Ma`]
    T : npt.NDArray[np.floating]
        Ambient temperature, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        True airspeed, [:math:`m \ s^{-1}`]

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    return mach_number * np.sqrt((constants.kappa * constants.R_d) * T)


def lbs_to_kg(lbs: ArrayScalarLike) -> ArrayScalarLike:
    r"""Convert mass from pounds (lbs) to kilograms (kg).

    Parameters
    ----------
    lbs : ArrayScalarLike
        mass, pounds [:math:`lbs`]

    Returns
    -------
    ArrayScalarLike
        mass, kilograms [:math:`kg`]
    """
    return lbs * 0.45359


def dt_to_seconds(
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating]:
    """Convert a time delta to seconds as a float with specified ``dtype`` precision.

    Parameters
    ----------
    dt : np.ndarray
        Time delta for each waypoint
    dtype : np.dtype
        Data type of the output array

    Returns
    -------
    np.ndarray
        Time delta in seconds as a float
    """
    out = np.empty(dt.shape, dtype=dtype)
    np.divide(dt, np.timedelta64(1, "s"), out=out)
    return out
