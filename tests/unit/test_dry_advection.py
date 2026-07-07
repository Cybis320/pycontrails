"""Test the :mod:`dry_advection` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from pycontrails import Flight, GeoVectorDataset, MetDataset
from pycontrails.models.cocip import Cocip
from pycontrails.models.cocip.contrail_properties import emulated_crystal_radius
from pycontrails.models.dry_advection import DryAdvection, MoistAdvection, MoistAdvectionParams
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.physics import geo, units


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


def test_dry_advection_downwash_displacement() -> None:
    """Wake-vortex downwash is a fixed descent at formation, independent of dt.

    In a still atmosphere (no wind, no vertical velocity) the only level change is
    the downwash. Enabling it sinks each waypoint once by the hydrostatic
    pressure-equivalent of ``downwash_distance``, and the descent is the same for any
    ``dt_integration`` — the property the previous per-step-velocity model lacked
    (which over-descended ~60x at the 30 min default).
    """
    lon0, lat0, lev0 = 0.0, 45.0, 250.0
    t_air = 220.0
    longitude = np.arange(lon0 - 1.0, lon0 + 1.01, 0.5)
    latitude = np.arange(lat0 - 1.0, lat0 + 1.01, 0.5)
    level = np.array([200.0, 250.0, 300.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")

    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    met["eastward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["northward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["lagrangian_tendency_of_air_pressure"] = xr.DataArray(
        np.zeros(met.shape), coords=met.coords
    )
    met["air_temperature"] = xr.DataArray(np.full(met.shape, t_air), coords=met.coords)
    met["geopotential"] = xr.DataArray(np.full(met.shape, 1.0e5), coords=met.coords)

    distance = 300.0
    # Expected one-time descent in hPa from the hydrostatic relation the model uses.
    rho = (lev0 * 100.0) / (287.05 * t_air)
    expected_drop = rho * 9.80665 * distance / 100.0

    def final_level(dt: np.timedelta64, apply: bool) -> float:
        src = GeoVectorDataset(longitude=[lon0], latitude=[lat0], level=[lev0], time=[time[0]])
        params = {
            "azimuth": None,
            "width": None,
            "depth": None,
            "max_age": np.timedelta64(1, "h"),
            "dt_integration": dt,
            "apply_downwash": apply,
            "downwash_distance": distance,
        }
        out = DryAdvection(met, params).eval(src)
        return float(out["level"][int(np.argmax(out["age"]))])

    # No downwash: the level is unchanged in a still atmosphere.
    assert final_level(np.timedelta64(30, "m"), apply=False) == pytest.approx(lev0, abs=1e-6)

    # Downwash: the parcel descends once by the expected hydrostatic amount...
    drop_30m = final_level(np.timedelta64(30, "m"), apply=True) - lev0
    drop_5m = final_level(np.timedelta64(5, "m"), apply=True) - lev0
    assert drop_30m == pytest.approx(expected_drop, abs=0.05)
    assert expected_drop == pytest.approx(11.6, abs=0.5)  # ~300 m at 250 hPa

    # ...and the descent is independent of the integration step.
    assert drop_30m == pytest.approx(drop_5m, abs=1e-6)


def test_dry_advection_wake_vortex_downwash() -> None:
    """Physics downwash (downwash_distance=None) computes a per-waypoint centroid from the met.

    The contrail-centroid sinking ``0.25 * dz_max`` is derived from the aircraft and met via
    ``wake_vortex.max_downward_displacement``, resolving wingspan/mass from the source (or the
    PS database by ``aircraft_type``, with mass defaulting to ``0.85 * MTOW``).
    """
    lon0, lat0 = 0.0, 45.0
    longitude = np.arange(lon0 - 1.0, lon0 + 1.01, 0.5)
    latitude = np.arange(lat0 - 1.0, lat0 + 1.01, 0.5)
    level = np.array([150.0, 200.0, 250.0, 300.0, 350.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")
    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    met["eastward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["northward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["lagrangian_tendency_of_air_pressure"] = xr.DataArray(
        np.zeros(met.shape), coords=met.coords
    )
    met["air_temperature"] = xr.DataArray(np.full(met.shape, 220.0), coords=met.coords)
    met["geopotential"] = xr.DataArray(np.full(met.shape, 1.0e5), coords=met.coords)

    df = pd.DataFrame(
        {
            "longitude": np.linspace(lon0 - 0.5, lon0 + 0.5, 8),
            "latitude": np.full(8, lat0),
            "altitude": np.full(8, 11000.0),
            "time": pd.date_range(time[0], time[0] + pd.Timedelta("20min"), periods=8),
        }
    )
    params = {
        "azimuth": None,
        "width": None,
        "depth": None,
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(10, "m"),
        "apply_downwash": True,
        "downwash_distance": None,  # physics: per-waypoint centroid
    }

    # Explicit aircraft params on the source.
    fl = Flight(
        df.copy(),
        attrs={
            "flight_id": "x",
            "wingspan": 60.0,
            "true_airspeed": 240.0,
            "aircraft_mass": 200000.0,
        },
    )
    out = DryAdvection(met, params).eval(fl)
    dz_max = out["downwash_dz_max"]
    assert np.all(np.isfinite(dz_max))
    assert np.all(dz_max > 0)
    assert 40.0 < float(np.nanmean(dz_max)) < 1000.0  # physical vortex-core sink [m]
    assert units.pl_to_m(out["level"]).min() < 11000.0  # the centroid descent was applied

    # PS fallback by aircraft_type (mass = 0.85 * MTOW, wingspan from the PS db).
    fl2 = Flight(
        df.copy(), attrs={"flight_id": "y", "aircraft_type": "A359", "true_airspeed": 240.0}
    )
    out2 = DryAdvection(met, params).eval(fl2)
    assert np.all(out2["downwash_dz_max"] > 0)

    # Missing aircraft info raises a clear error.
    fl3 = Flight(df.copy(), attrs={"flight_id": "z", "true_airspeed": 240.0})
    with pytest.raises(ValueError, match="wingspan"):
        DryAdvection(met, params).eval(fl3)


def test_dry_advection_downwash_time_resolved() -> None:
    """At fine dt the wake downwash descent is resolved step-by-step, not a single jump.

    The descent is applied as the per-step increment z_c(age2) - z_c(age1), so with dense
    timesteps the first minutes show a progressive (decelerating) descent rather than the
    full sink in one step.
    """
    lon0, lat0 = 0.0, 45.0
    longitude = np.arange(lon0 - 1.0, lon0 + 1.01, 0.5)
    latitude = np.arange(lat0 - 1.0, lat0 + 1.01, 0.5)
    level = np.array([150.0, 200.0, 250.0, 300.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")
    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    for k in ("eastward_wind", "northward_wind", "lagrangian_tendency_of_air_pressure"):
        met[k] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["air_temperature"] = xr.DataArray(np.full(met.shape, 220.0), coords=met.coords)
    met["geopotential"] = xr.DataArray(np.full(met.shape, 1.0e5), coords=met.coords)

    src = GeoVectorDataset(longitude=[lon0], latitude=[lat0], level=[240.0], time=[time[0]])
    src.attrs.update(wingspan=60.0, true_airspeed=240.0, aircraft_mass=200000.0)
    params = {
        "azimuth": None,
        "width": None,
        "depth": None,
        "max_age": np.timedelta64(6, "m"),
        "dt_integration": np.timedelta64(15, "s"),  # fine: resolve the wake phase
        "apply_downwash": True,
        "downwash_distance": None,
    }
    out = DryAdvection(met, params).eval(src)
    d = out.dataframe.sort_values("age")
    drops = -np.diff(units.pl_to_m(d["level"].to_numpy()))  # descent per step [m], positive down

    assert np.all(drops > -1e-6)  # monotone descent
    assert (drops > 0.5).sum() >= 3  # spread over several steps (time-resolved, not one-shot)
    assert drops[0] < 0.9 * drops.sum()  # not all descent in the first step
    assert drops[-1] < drops[0]  # decelerating


def test_moist_advection_defaults() -> None:
    """MoistAdvection is a DryAdvection preset with the wet physics on in centerline mode."""
    assert issubclass(MoistAdvection, DryAdvection)
    p = MoistAdvectionParams()
    # Wake downwash (physics, per-waypoint) and RHi-modulated sedimentation on by default.
    assert p.apply_downwash is True
    assert p.downwash_distance is None
    assert p.microphysical_sedimentation is True
    # Pointwise centerline (no wind-shear geometry).
    assert p.azimuth is None
    assert p.width is None
    assert p.depth is None
    assert MoistAdvection.default_params is MoistAdvectionParams


def test_dry_advection_sedimentation_density_aware() -> None:
    """Sedimentation sinks the plume at the hydrostatic rate rho*g*v_fall.

    In a still atmosphere the only level change is sedimentation. The pressure
    descent over a step equals rho*g*v_fall*dt, so -- unlike a fixed Pa/s knob -- a
    fixed fall speed sinks a denser, lower-altitude parcel faster in pressure.
    """
    t_air = 220.0
    lon0, lat0 = 0.0, 45.0
    longitude = np.arange(lon0 - 1.0, lon0 + 1.01, 0.5)
    latitude = np.arange(lat0 - 1.0, lat0 + 1.01, 0.5)
    # Wide level range so sedimenting parcels stay inside the met pressure domain.
    level = np.array([150.0, 200.0, 250.0, 300.0, 350.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")

    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    met["eastward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["northward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["lagrangian_tendency_of_air_pressure"] = xr.DataArray(
        np.zeros(met.shape), coords=met.coords
    )
    met["air_temperature"] = xr.DataArray(np.full(met.shape, t_air), coords=met.coords)
    met["geopotential"] = xr.DataArray(np.full(met.shape, 1.0e5), coords=met.coords)

    v_fall = 0.1  # m/s
    dt = np.timedelta64(10, "m")
    dt_s = dt / np.timedelta64(1, "s")

    def one_step_drop(level0: float, velocity: float) -> float:
        src = GeoVectorDataset(longitude=[lon0], latitude=[lat0], level=[level0], time=[time[0]])
        params = {
            "azimuth": None,
            "width": None,
            "depth": None,
            "max_age": dt,
            "dt_integration": dt,
            "sedimentation_velocity": velocity,
        }
        out = DryAdvection(met, params).eval(src)
        return float(out["level"][int(np.argmax(out["age"]))]) - level0

    # No sedimentation: the level is unchanged in a still atmosphere.
    assert one_step_drop(250.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    # The descent over one step equals the hydrostatic rho*g*v*dt at the parcel level.
    for lev0 in (200.0, 250.0, 300.0):
        rho = (lev0 * 100.0) / (287.05 * t_air)
        expected = rho * 9.80665 * v_fall * dt_s / 100.0  # hPa
        assert one_step_drop(lev0, v_fall) == pytest.approx(expected, rel=1e-3)

    # Density-aware: a denser parcel (300 hPa) sinks 1.5x faster in pressure than a
    # 200 hPa parcel for the same fall speed -- a fixed Pa/s rate would give 1.0.
    ratio = one_step_drop(300.0, v_fall) / one_step_drop(200.0, v_fall)
    assert ratio == pytest.approx(1.5, rel=1e-3)


def test_emulated_crystal_radius_grows_monotonically() -> None:
    """The CoCiP-surrogate crystal radius grows monotonically with age at micron scale."""
    age_s = np.array([0.0, 600.0, 1800.0, 3600.0, 10800.0])  # 0, 10, 30, 60, 180 min
    r = emulated_crystal_radius(age_s)
    assert np.all(np.diff(r) > 0)  # monotone growth
    assert np.all((r > 0.5e-6) & (r < 50e-6))  # microns, physical for young contrails

    # RHi modulation: more supersaturation -> larger crystals, and it changes the age-only value.
    r_low = emulated_crystal_radius(age_s, rhi=np.full_like(age_s, 1.1))
    r_high = emulated_crystal_radius(age_s, rhi=np.full_like(age_s, 1.5))
    assert np.all(r_high > r_low)
    assert np.all(r_low != r)


def test_dry_advection_microphysical_sedimentation_curves() -> None:
    """Microphysical sedimentation gives a CoCiP-like *curving* descent (growing crystals).

    In a still atmosphere the only level change is sedimentation. With
    ``microphysical_sedimentation`` the fall speed tracks the age-growing crystal radius,
    so the descent accelerates with age -- unlike a constant fall speed. Also confirmed:
    it descends further than the (kinematic) no-sedimentation run.
    """
    t_air = 220.0
    lon0, lat0 = 0.0, 45.0
    longitude = np.arange(lon0 - 1.0, lon0 + 1.01, 0.5)
    latitude = np.arange(lat0 - 1.0, lat0 + 1.01, 0.5)
    level = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")
    met = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)
    met["eastward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["northward_wind"] = xr.DataArray(np.zeros(met.shape), coords=met.coords)
    met["lagrangian_tendency_of_air_pressure"] = xr.DataArray(
        np.zeros(met.shape), coords=met.coords
    )
    met["air_temperature"] = xr.DataArray(np.full(met.shape, t_air), coords=met.coords)
    # Supersaturated (RHi ~ 150% near 200 hPa) so the RHi-modulated crystal growth is active.
    met["specific_humidity"] = xr.DataArray(np.full(met.shape, 1.3e-4), coords=met.coords)
    met["geopotential"] = xr.DataArray(np.full(met.shape, 1.0e5), coords=met.coords)

    src = GeoVectorDataset(longitude=[lon0], latitude=[lat0], level=[200.0], time=[time[0]])
    params = {
        "azimuth": None,
        "width": None,
        "depth": None,
        "max_age": np.timedelta64(3, "h"),
        "dt_integration": np.timedelta64(20, "m"),
        "microphysical_sedimentation": True,
    }
    out = DryAdvection(met, params).eval(src)
    levels = out.dataframe.sort_values("age")["level"].to_numpy()

    # Descends monotonically (pressure increases) and the per-step drop grows with age
    # (crystals grow -> faster fall) -- the curving signature a constant knob lacks.
    drops = np.diff(levels)
    assert np.all(drops > 0)
    assert drops[-1] > 2.0 * drops[0]

    # Descends further than the kinematic (no-sedimentation) run, which is unchanged in a
    # still atmosphere (the RHi modulation shrinks the crystals, so the descent is modest).
    kinematic = DryAdvection(met, {**params, "microphysical_sedimentation": False}).eval(src)
    assert levels[-1] > kinematic.dataframe.sort_values("age")["level"].to_numpy()[-1] + 0.3
