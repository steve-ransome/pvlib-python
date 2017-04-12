from datetime import datetime, timedelta
import inspect
from math import isnan
from pytz import timezone

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose

from conftest import requires_siphon, has_siphon, skip_windows

pytestmark = pytest.mark.skipif(not has_siphon, reason='requires siphon')

from pvlib.location import Location

if has_siphon:
    import requests
    from requests.exceptions import HTTPError
    from xml.etree.ElementTree import ParseError

    from pvlib.forecast import GFS, HRRR_ESRL, HRRR, NAM, NDFD, RAP

    # setup times and location to be tested. Tucson, AZ
    _latitude = 32.2
    _longitude = -110.9
    _tz = 'US/Arizona'
    _start = pd.Timestamp.now(tz=_tz)
    _end = _start + pd.Timedelta(days=1)
    _modelclasses = [
        GFS, NAM, HRRR, NDFD, RAP,
        skip_windows(
            pytest.mark.xfail(
                pytest.mark.timeout(HRRR_ESRL, timeout=60),
                reason="HRRR_ESRL is unreliable"))]
    _working_models = []
    _variables = ['temp_air', 'wind_speed', 'total_clouds', 'low_clouds',
                  'mid_clouds', 'high_clouds', 'dni', 'dhi', 'ghi',]
    _nonnan_variables = ['temp_air', 'wind_speed', 'total_clouds', 'dni',
                         'dhi', 'ghi',]
else:
    _modelclasses = []


# make a model object for each model class
# get the data for that model and store it in an
# attribute for further testing
@requires_siphon
@pytest.fixture(scope='module', params=_modelclasses)
def model(request):
    amodel = request.param()
    amodel.raw_data = \
        amodel.get_data(_latitude, _longitude, _start, _end)
    return amodel


@requires_siphon
def test_process_data(model):
    for how in ['liujordan', 'clearsky_scaling']:
        data = model.process_data(model.raw_data, how=how)
        for variable in _nonnan_variables:
            assert not data[variable].isnull().values.any()


@requires_siphon
def test_vert_level():
    amodel = NAM()
    vert_level = 5000
    data = amodel.get_processed_data(_latitude, _longitude, _start, _end,
                                     vert_level=vert_level)

@requires_siphon
def test_datetime():
    amodel = NAM()
    start = datetime.now()
    end = start + timedelta(days=1)
    data = amodel.get_processed_data(_latitude, _longitude , start, end)


@requires_siphon
def test_queryvariables():
    amodel = GFS()
    old_variables = amodel.variables
    new_variables = ['u-component_of_wind_height_above_ground']
    data = amodel.get_data(_latitude, _longitude, _start, _end,
                           query_variables=new_variables)
    data['u-component_of_wind_height_above_ground']


@requires_siphon
def test_latest():
    GFS(set_type='latest')


@requires_siphon
def test_full():
    GFS(set_type='full')


@requires_siphon
def test_temp_convert():
    amodel = GFS()
    data = pd.DataFrame({'temp_air': [273.15]})
    data['temp_air'] = amodel.kelvin_to_celsius(data['temp_air'])

    assert_allclose(data['temp_air'].values, 0.0)


# @requires_siphon
# def test_bounding_box():
#     amodel = GFS()
#     latitude = [31.2,32.2]
#     longitude = [-111.9,-110.9]
#     new_variables = {'temperature':'Temperature_surface'}
#     data = amodel.get_query_data(latitude, longitude, _start, _end,
#                                  variables=new_variables)


@requires_siphon
def test_set_location():
    amodel = GFS()
    latitude, longitude = 32.2, -110.9
    time = datetime.now(timezone('UTC'))
    amodel.set_location(time, latitude, longitude)


def test_cloud_cover_to_transmittance_linear():
    amodel = GFS()
    assert_allclose(amodel.cloud_cover_to_transmittance_linear(0), 0.75)
    assert_allclose(amodel.cloud_cover_to_transmittance_linear(100), 0.0)


def test_cloud_cover_to_ghi_linear():
    amodel = GFS()
    ghi_clear = 1000
    offset = 25
    out = amodel.cloud_cover_to_ghi_linear(0, ghi_clear, offset=offset)
    assert_allclose(out, 1000)
    out = amodel.cloud_cover_to_ghi_linear(100, ghi_clear, offset=offset)
    assert_allclose(out, 250)
