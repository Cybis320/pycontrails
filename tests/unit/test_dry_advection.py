"""Test the :mod:`dry_advection` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from pycontrails import Flight, GeoVectorDataset, MetDataset
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.physics import geo


@pytest.fixture()
def source(met_cocip1: MetDataset) -> GeoVectorDataset:
    """Return a GeoVectorDataset."""

    ds = met_cocip1.data.isel(time=[0])
    mds = MetDataset(ds.drop_vars(ds.data_vars))
    return mds.to_vector()


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, 270.0, None])
def test_dry_advection(
    met_cocip1: MetDataset, source: GeoVectorDataset, azimuth: float | None
) -> None:
    """Test the :class:`DryAdvection` model."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "azimuth": azimuth,
    }
    if azimuth is None:
        params["width"] = None
        params["depth"] = None

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)
    assert len(out) == 3524

    assert out["age"].max() == np.timedelta64(1, "h")

    if azimuth is None:
        assert len(out.data) == 10
    else:
        assert len(out.data) == 16

    # Pin some values to ensure that the model is working as expected
    abs = 0.1
    if azimuth in (0.0, 180.0):
        assert np.nanmean(out["width"]) == pytest.approx(1019.6, abs=abs)
    elif azimuth in (90.0, 270.0):
        assert np.nanmean(out["width"]) == pytest.approx(763.9, abs=abs)
    else:
        assert "width" not in out


@pytest.mark.filterwarnings("ignore")
def test_compare_dry_advection_to_cocip(
    flight_cocip1: Flight,
    met_cocip1: MetDataset,
    rad_cocip1: MetDataset,
) -> None:
    """Compare the dry advection model to Cocip predictions."""

    params = {"max_age": np.timedelta64(1, "h"), "dt_integration": np.timedelta64(5, "m")}

    model = DryAdvection(met_cocip1, params)
    out = model.eval(flight_cocip1)
    df1 = out.dataframe
    assert df1["longitude"].notna().all()
    assert df1["latitude"].notna().all()
    assert df1["level"].notna().all()
    assert df1["time"].notna().all()

    assert df1.shape == (208, 17)
    df1_sl = df1.loc[df1["time"] == "2019-01-01T01:25"]

    model = Cocip(
        met_cocip1,
        rad_cocip1,
        params,
        filter_sac=False,
        filter_initially_persistent=False,
        humidity_scaling=ConstantHumidityScaling(rhi_adj=0.6),
    )
    model.eval(flight_cocip1)
    df2 = model.contrail
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (196, 58)
    df2_sl = df2[df2["time"] == "2019-01-01T01:25"]

    # Pin some mean values to demonstrate the difference in vertical advection
    assert df1_sl["level"].mean() == pytest.approx(219.56, abs=0.01)
    assert df2_sl["level"].mean() == pytest.approx(222.07, abs=0.1)


@pytest.mark.parametrize(
    ("verbose_outputs", "include_source_in_output"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dry_advection_verbose_outputs(
    met_cocip1: MetDataset,
    source: GeoVectorDataset,
    verbose_outputs: bool,
    include_source_in_output: bool,
) -> None:
    """Test the :class:`DryAdvection` model."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "verbose_outputs": verbose_outputs,
        "include_source_in_output": include_source_in_output,
    }

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)
    n_variables = len(out.data)
    n_rows = len(out)

    extra_keys = ("ds_dz", "dsn_dz", "dT_dz")
    if verbose_outputs:
        assert all(key in out for key in extra_keys)
        assert n_variables == 19
    else:
        assert not any(key in out for key in extra_keys)
        assert n_variables == 16

    if include_source_in_output:
        assert n_rows == 3524 + len(source)
    else:
        assert n_rows == 3524


@pytest.mark.parametrize("include_source_in_output", [True, False])
def test_dry_advection_flight_id_in_output(
    met_cocip1: MetDataset, source: GeoVectorDataset, include_source_in_output: bool
) -> None:
    """Test the inclusion of a ``flight_id`` column in :class:`DryAdvection` output."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "include_source_in_output": include_source_in_output,
    }

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)

    assert "flight_id" not in source
    assert "flight_id" not in out

    source1 = source.filter(source["latitude"] < 55.0, copy=True)
    source2 = source.filter(source["latitude"] >= 55, copy=True)
    source1["flight_id"] = np.full(len(source1), "flight1")
    source2["flight_id"] = np.full(len(source2), "flight2")
    source = source1 + source2
    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)

    assert "flight_id" in source
    assert "flight_id" in out
    if include_source_in_output:
        assert (out["flight_id"] == "flight1").sum() == 2054 + len(source1)
        assert (out["flight_id"] == "flight2").sum() == 1470 + len(source2)
    else:
        assert (out["flight_id"] == "flight1").sum() == 2054
        assert (out["flight_id"] == "flight2").sum() == 1470


def test_dry_advection_gap_in_waypoints(met_cocip1: MetDataset) -> None:
    """Confirm fix for bug in which a large temporal gap between waypoints broke implementation."""
    params = {
        "max_age": np.timedelta64(20, "m"),
        "dt_integration": np.timedelta64(1, "m"),
    }

    t0 = met_cocip1.data["time"].isel(time=0).values
    t1 = t0 + np.timedelta64(30, "m")

    model = DryAdvection(met_cocip1, params)
    source = GeoVectorDataset(
        longitude=[-30.0, -30.0],
        latitude=[55.0, 55.0],
        level=[250.0, 250.0],
        time=[t0, t1],
    )
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)

    # The output should have 20 times for waypoint 1 and 20 times for waypoint 2
    waypoint0 = out.dataframe.query("waypoint == 0")
    expected0 = pd.date_range(t0, periods=21, freq="1min", inclusive="right").to_series(name="time")
    pd.testing.assert_series_equal(waypoint0["time"], expected0, check_index=False)

    waypoint1 = out.dataframe.query("waypoint == 1")
    expected1 = pd.date_range(t1, periods=21, freq="1min", inclusive="right").to_series(name="time")
    pd.testing.assert_series_equal(waypoint1["time"], expected1, check_index=False)


def test_dry_advection_with_timesteps(
    met_cocip1: MetDataset, flight_cocip1: GeoVectorDataset
) -> None:
    """Test the :class:`DryAdvection` model with manually-specified timesteps."""
    timesteps = np.arange(
        flight_cocip1["time"].min() + np.timedelta64(5, "m"),
        flight_cocip1["time"].max() - np.timedelta64(2, "m"),
        np.timedelta64(5, "m"),
    )

    model = DryAdvection(met_cocip1, max_age=None)
    with pytest.raises(ValueError, match="Timesteps must be set"):
        model.eval(flight_cocip1)

    model = DryAdvection(met_cocip1, max_age=None, timesteps=timesteps)
    out = model.eval(flight_cocip1)

    assert isinstance(out, GeoVectorDataset)
    assert len(out) == 158
    # the very first waypoint blows out of bounds, so it's not in the output
    np.testing.assert_array_equal(np.unique(out["time"]), timesteps[1:])

    model = DryAdvection(met_cocip1, max_age=np.timedelta64(10, "m"), timesteps=timesteps)
    out = model.eval(flight_cocip1)

    assert isinstance(out, GeoVectorDataset)
    assert len(out) == 35
    assert out["age"].max() <= np.timedelta64(10, "m")


def test_dry_advection_rk4_solid_body_rotation() -> None:
    """RK4 centerline stays accurate in curving flow (acceptance test for Fix 1).

    A solid-body rotation wind field (tangential speed 50 m/s at radius 100 km)
    continuously curves the wind, so forward Euler accrues large truncation error.
    Classical RK4 at a 5 min step must track the rotation to within 3 m at 3 h,
    measured against a converged 30 s reference integration of the same ODE. The
    ellipsoidal geometry cancels in this comparison, isolating integrator error.
    """
    speed = 50.0  # m/s
    radius = 100_000.0  # m
    omega = speed / radius  # rad/s
    lon0, lat0 = 0.0, 45.0

    # Local metres-per-degree at the rotation centre (WGS-84).
    m_per_deg_lon = (
        np.deg2rad(1.0)
        * float(geo.prime_vertical_radius_of_curvature(lat0))
        * np.cos(np.deg2rad(lat0))
    )
    m_per_deg_lat = np.deg2rad(1.0) * float(geo.meridional_radius_of_curvature(lat0))

    # Grid covering the orbit (~0.9 deg radius) with margin. The wind is linear in
    # position, so linear interpolation is exact at any resolution.
    longitude = np.arange(lon0 - 2.0, lon0 + 2.01, 0.25)
    latitude = np.arange(lat0 - 2.0, lat0 + 2.01, 0.25)
    level = np.array([250.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")

    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    lon_g, lat_g = np.meshgrid(
        met.data["longitude"].values, met.data["latitude"].values, indexing="ij"
    )
    dx = (lon_g - lon0) * m_per_deg_lon
    dy = (lat_g - lat0) * m_per_deg_lat
    u = -omega * dy  # solid-body rotation, counter-clockwise
    v = omega * dx

    def _grid(field_2d: np.ndarray) -> xr.DataArray:
        data = np.broadcast_to(field_2d[:, :, None, None], met.shape).astype("float64")
        return xr.DataArray(data, coords=met.coords)

    met["eastward_wind"] = _grid(u)
    met["northward_wind"] = _grid(v)
    met["lagrangian_tendency_of_air_pressure"] = _grid(np.zeros_like(u))
    # Required by the model but unused by the pointwise centerline; constant values.
    met["air_temperature"] = _grid(np.full_like(u, 220.0))
    met["geopotential"] = _grid(np.full_like(u, 1.0e5))

    # Start one radius east of the centre (initial wind is purely northward).
    source = GeoVectorDataset(
        longitude=[lon0 + radius / m_per_deg_lon],
        latitude=[lat0],
        level=[250.0],
        time=[time[0]],
    )

    params = {
        "azimuth": None,
        "width": None,
        "depth": None,
        "max_age": np.timedelta64(3, "h"),
        "dt_integration": np.timedelta64(5, "m"),
    }
    coarse = DryAdvection(met, params).eval(source)
    fine = DryAdvection(met, {**params, "dt_integration": np.timedelta64(30, "s")}).eval(source)

    def _endpoint(out: GeoVectorDataset) -> tuple[float, float]:
        i = int(np.argmax(out["age"]))
        return float(out["longitude"][i]), float(out["latitude"][i])

    lon_c, lat_c = _endpoint(coarse)
    lon_f, lat_f = _endpoint(fine)

    geod = pyproj.Geod(ellps="WGS84")
    _, _, error_m = geod.inv(lon_c, lat_c, lon_f, lat_f)
    assert error_m <= 3.0

    # Sanity: the parcel actually orbited rather than drifting off in a straight line
    # (a straight 50 m/s path would be ~540 km from the centre after 3 h).
    _, _, orbit_radius = geod.inv(lon0, lat0, lon_f, lat_f)
    assert 0.8 * radius < orbit_radius < 1.2 * radius
