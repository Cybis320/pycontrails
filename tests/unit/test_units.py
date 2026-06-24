"""Test pycontrails.physics.units module."""

from __future__ import annotations

from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.physics import constants, units


@pytest.fixture(scope="module")
def rng():
    """Get random number generator."""
    return np.random.default_rng(12345)


def test_ft_m(rng):
    """Check `ft_to_m` and `m_to_ft` are bijective."""
    ft1 = rng.uniform(0, 10000, 10000)

    m1 = units.ft_to_m(ft1)
    ft2 = units.m_to_ft(m1)
    np.testing.assert_array_almost_equal(ft1, ft2)

    m2 = units.ft_to_m(ft2)
    np.testing.assert_array_almost_equal(m1, m2)


def test_knots_mps(rng):
    """Check that `m_per_s_to_knots` and `knots_to_m_per_s` are bijective."""
    mps1 = rng.uniform(0, 10000, 10000)

    knots1 = units.m_per_s_to_knots(mps1)
    mps2 = units.knots_to_m_per_s(knots1)
    np.testing.assert_array_almost_equal(mps1, mps2)

    knots2 = units.m_per_s_to_knots(mps2)
    np.testing.assert_array_almost_equal(knots1, knots2)


def test_rad_deg():
    """Check `degrees_to_radians` conversion on a few values."""
    d = 0
    r = units.degrees_to_radians(d)
    assert r == 0

    d = 180
    r = units.degrees_to_radians(d)
    assert r == pytest.approx(3.14159265)


def test_rad_deg_agree_np(rng):
    """Check that pycontrails implementation of agrees with numpy implementation."""
    x = rng.uniform(0, 1000, 10000)

    rad = units.radians_to_degrees(x)
    np_rad = np.rad2deg(x)
    np.testing.assert_array_equal(rad, np_rad)

    deg = units.degrees_to_radians(x)
    np_deg = np.deg2rad(x)
    np.testing.assert_array_equal(deg, np_deg)


def test_m_pl_bijective():
    """Check that the functions `pl_to_m` and `m_to_pl` are bijective."""
    m1 = np.arange(0, 50000)
    m2 = units.pl_to_m(units.m_to_pl(m1))
    np.testing.assert_allclose(m1, m2, atol=1e-11)

    pl1 = np.arange(1, 2000)
    pl2 = units.m_to_pl(units.pl_to_m(pl1))
    np.testing.assert_allclose(pl1, pl2, atol=1e-11)


def classical_pl_to_m(pl):
    """Classical implementation of pl_to_m."""
    pl_pa = pl * 100
    return (constants.T_msl / 0.0065) * (1 - (pl_pa / constants.p_surface) ** (1 / 5.255))


def test_pl_to_m_close_to_classical(rng):
    """Check that the classical conversion agrees for low altitudes."""
    p = rng.uniform(200, 1000, 10000)
    m1 = classical_pl_to_m(p)
    m2 = units.pl_to_m(p)
    np.testing.assert_allclose(m1, m2, rtol=1e-3)


def test_m_to_pl_int_and_array():
    """Check vectorized call agrees with naive loop.

    Calling two functions that use `np.piecewise` pattern.
    """
    arr = np.arange(15000)
    y1 = units.m_to_pl(arr)
    y2 = [units.m_to_pl(x) for x in arr]
    np.testing.assert_array_equal(y1, y2)

    arr = np.arange(100, 1000)
    y1 = units.pl_to_m(arr)
    y2 = [units.pl_to_m(x) for x in arr]
    np.testing.assert_array_equal(y1, y2)


def test_mach_tas(rng: np.random.Generator):
    """Check that the functions `tas_to_mach_number` and `mach_number_to_tas` are bijective."""
    T = rng.uniform(200, 300, 10000)
    tas1 = rng.uniform(200, 300, 10000)
    ma1 = units.tas_to_mach_number(tas1, T)

    # Ensure somewhat realistic values
    assert np.all(np.isfinite(ma1))
    assert np.all(ma1 < 1.1)
    assert np.all(ma1 > 0.5)
    assert ma1.mean() == pytest.approx(0.79, abs=0.01)

    tas2 = units.mach_number_to_tas(ma1, T)
    np.testing.assert_array_almost_equal(tas1, tas2, decimal=10)

    ma2 = units.tas_to_mach_number(tas2, T)
    assert np.all(np.isfinite(ma2))
    np.testing.assert_array_almost_equal(ma1, ma2, decimal=10)


@pytest.mark.parametrize(
    "func",
    [
        func
        for name, func in getmembers(units, isfunction)
        if name
        not in [
            "support_arraylike",
            "longitude_distance_to_m",
            "m_to_longitude_distance",
            "tas_to_mach_number",
            "mach_number_to_tas",
            "geopotential_to_geometric_height",
            "dt_to_seconds",
        ]
        and not name.startswith("_")
    ],
)
def test_arraylike_support(func):
    """Check that `unit` module functions support ArrayLike parameters."""
    x = 123
    y = 123.456789
    assert isinstance(func(x), float)
    assert isinstance(func(y), float)

    z = np.array([x, y])
    assert isinstance(func(z), np.ndarray)

    # Some functions can handle Series or DataArrays
    # Others convert to numpy
    da = xr.DataArray(z)
    assert isinstance(func(da), xr.DataArray | np.ndarray)

    # Calling xr.apply_ufunc gives same result
    xr.testing.assert_equal(func(da), xr.apply_ufunc(func, da))

    s = pd.Series(z)
    assert isinstance(func(s), pd.Series | np.ndarray)


@pytest.mark.parametrize("func", [units.pl_to_m, units.m_to_pl])
def test_handle_nan(func):
    """Check that `unit` module functions using `np.piecewise` pass NaN values through."""
    x = np.array([100, 200, np.nan, 300], dtype=np.float64)
    y = func(x)
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(np.isfinite(y), [True, True, False, True])
    assert np.isnan(x[2])
    assert np.isnan(y[2])


def test_normal_gravity_reference_values():
    """Check WGS-84 Somigliana normal gravity against known reference values."""
    assert units.normal_gravity(0.0) == pytest.approx(9.78033, abs=1e-4)
    assert units.normal_gravity(45.0) == pytest.approx(9.80620, abs=1e-4)
    assert units.normal_gravity(52.0) == pytest.approx(9.81247, abs=1e-4)
    assert units.normal_gravity(90.0) == pytest.approx(9.83218, abs=1e-4)

    # Gravity increases monotonically from the equator to the pole.
    g = units.normal_gravity(np.array([0.0, 45.0, 52.0, 90.0]))
    assert np.all(np.diff(g) > 0.0)

    # Symmetric about the equator.
    np.testing.assert_allclose(
        units.normal_gravity(np.array([-45.0, -10.0, 30.0])),
        units.normal_gravity(np.array([45.0, 10.0, -30.0])),
    )


def test_geopotential_to_geometric_height_reference():
    """Check geopotential to geometric height conversion across latitudes.

    The input geopotential corresponds to a WMO geopotential height of 11000 m
    (``Phi = 11000 * constants.g``). The latitude-correct geometric height is
    larger than 11000 m near the equator and smaller near the pole.
    """
    geopotential = 11000.0 * constants.g

    z_equator = units.geopotential_to_geometric_height(geopotential, 0.0)
    z_mid = units.geopotential_to_geometric_height(geopotential, 45.0)
    z_pole = units.geopotential_to_geometric_height(geopotential, 90.0)

    assert z_equator == pytest.approx(11048.8, abs=0.5)
    assert z_mid == pytest.approx(11019.1, abs=0.5)
    assert z_pole == pytest.approx(10990.0, abs=0.5)

    # The latitude-systematic correction spans ~59 m from equator to pole, with the
    # equator height under-stated and the pole height over-stated by the legacy
    # fixed-``g0`` diagnostic.
    assert z_equator - z_pole == pytest.approx(59.0, abs=1.0)
    assert z_equator > z_mid > z_pole


def test_geopotential_to_geometric_height_crossover():
    """Near 45 deg latitude the result matches the legacy fixed-``g0`` height.

    45 deg is the crossover latitude where the WGS-84 normal gravity is closest to
    the WMO constant ``constants.g``, so the latitude-correct geometric height
    agrees with the legacy ``Phi / g0`` curvature formula to within ~0.5 m (versus a
    ~30 m deviation at the equator and pole).
    """
    geopotential = 11000.0 * constants.g

    # Legacy geometric height: divide by the fixed WMO g0 and apply the curvature
    # correction with a fixed effective radius.
    h_legacy = geopotential / constants.g  # == 11000 m
    r_fixed = 6356766.0
    legacy = h_legacy + h_legacy**2 / (r_fixed - h_legacy)

    result = units.geopotential_to_geometric_height(geopotential, 45.0)
    assert result == pytest.approx(legacy, abs=1.0)

    # The deviation at the equator is far larger, confirming 45 deg is the crossover.
    equator = units.geopotential_to_geometric_height(geopotential, 0.0)
    assert abs(equator - legacy) > 25.0


def test_geopotential_to_geometric_height_broadcast_dataarray():
    """Latitude broadcasts against a geopotential field while preserving type."""
    geopotential = 11000.0 * constants.g
    latitudes = np.array([0.0, 45.0, 90.0])

    # 2D geopotential field (e.g. the ERA5 ``z`` field) and a latitude coordinate.
    z = xr.DataArray(
        np.full((3, 4), geopotential),
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": np.arange(4.0)},
    )

    out = units.geopotential_to_geometric_height(z, z["latitude"])

    # Type is preserved: a DataArray is not silently downcast to a bare ndarray.
    assert isinstance(out, xr.DataArray)
    assert out.shape == z.shape
    assert set(out.dims) == set(z.dims)

    # Each latitude row agrees with the scalar computation at that latitude.
    for i, lat in enumerate(latitudes):
        expected = units.geopotential_to_geometric_height(geopotential, float(lat))
        np.testing.assert_allclose(out.isel(latitude=i).to_numpy(), expected)
