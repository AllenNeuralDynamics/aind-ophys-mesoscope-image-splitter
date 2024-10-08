import pytest
import copy
import pathlib
import numpy as np
from unittest.mock import patch

from mesoscope_file_splitter.splitter import (
    mkstemp_clean,
    ZStackSplitter
    )


@pytest.fixture(scope='session')
def baseline_zstack_metadata():
    """
    A skeleton on which to build various ill-formed ZStack metadata
    instances
    """
    n_rois = 3

    main_roi_list = []
    for ii in range(n_rois):
        main_roi_list.append({'zs': -10,  # for z_stack, ;zs' does not matter
                              'discretePlaneMode': 1})

    main_roi_metadata = {'RoiGroups':
                         {'imagingRoiGroup':
                          {'rois': main_roi_list}}}

    z0_values = np.linspace(1.0, 3.0, 13)
    z1_values = np.linspace(3.0, 5.0, 13)
    zz = []
    for z0, z1 in zip(z0_values, z1_values):
        zz.append([z0, z1])
    return [{'SI.hStackManager.zsAllActuators': zz,
             'SI.hChannels.channelSave': [1, 2]},
            main_roi_metadata]


@pytest.fixture(scope='session')
def z_stack_many_discrete_plane_zero(tmp_path_factory,
                                     baseline_zstack_metadata):
    """
    Create a set of z_stacks in which multiple ROIs
    are marked with discretePlaneMode==0
    """

    path_to_metadata = dict()
    tmp_dir = tmp_path_factory.mktemp('z_stack_many')
    for ii in range(3):
        tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.tiff')
        metadata = copy.deepcopy(baseline_zstack_metadata)
        rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
        if ii == 0:
            rois[0]['discretePlaneMode'] = 0
            rois[1]['discretePlaneMode'] = 0
        else:
            rois[ii]['discretePlaneMode'] = 0

        path_to_metadata[tmp_path] = metadata

    return path_to_metadata


def test_many_discrete_planes(
        z_stack_many_discrete_plane_zero):
    """
    Test that an error is raised if more than one ROI is marked
    with discretePlaneMode==0
    """
    z_stack_path_to_metadata = z_stack_many_discrete_plane_zero

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    tiff_path_list = [pathlib.Path(n)
                      for n in z_stack_path_to_metadata.keys()]

    to_replace = 'mesoscope_file_splitter.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace, new=mock_read_metadata):
        with pytest.raises(RuntimeError,
                           match="More than one ROI has discrete"):
            ZStackSplitter(tiff_path_list=tiff_path_list)


@pytest.fixture(scope='session')
def z_stack_no_discrete_plane_zero(tmp_path_factory,
                                   baseline_zstack_metadata):
    """
    Create a set of z_stacks in which multiple ROIs
    are marked with discretePlaneMode==0
    """

    path_to_metadata = dict()
    tmp_dir = tmp_path_factory.mktemp('z_stack_none')
    for ii in range(3):
        tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.tiff')
        metadata = copy.deepcopy(baseline_zstack_metadata)
        rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
        if ii != 1:
            rois[ii]['discretePlaneMode'] = 0

        path_to_metadata[tmp_path] = metadata

    return path_to_metadata


def test_no_discrete_planes(
        z_stack_no_discrete_plane_zero):
    """
    Test that an error is raised if no ROI is marked
    with discretePlaneMode==0
    """
    z_stack_path_to_metadata = z_stack_no_discrete_plane_zero

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    tiff_path_list = [pathlib.Path(n)
                      for n in z_stack_path_to_metadata.keys()]

    to_replace = 'mesoscope_file_splitter.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace, new=mock_read_metadata):
        with pytest.raises(RuntimeError,
                           match="Could not find discrete"):
            ZStackSplitter(tiff_path_list=tiff_path_list)


@pytest.fixture(scope='session')
def z_stack_z_array_odd_shape(tmp_path_factory,
                              baseline_zstack_metadata):
    """
    Create a set of z_stacks in which z_array is not (N, 2)
    """

    path_to_metadata = dict()
    tmp_dir = tmp_path_factory.mktemp('z_stack_none')
    for ii in range(3):
        tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.tiff')
        metadata = copy.deepcopy(baseline_zstack_metadata)
        rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
        rois[ii]['discretePlaneMode'] = 0

        if ii == 1:
            key_name = 'SI.hStackManager.zsAllActuators'
            metadata[0].pop(key_name)
            metadata[0][key_name] = [[1, 2, 3], [1, 2, 3]]

        path_to_metadata[tmp_path] = metadata

    return path_to_metadata


def test_z_odd_shape(
        z_stack_z_array_odd_shape):
    """
    Test that an error is raised if the z_array in the metadata
    is not (N, 2)
    """
    z_stack_path_to_metadata = z_stack_z_array_odd_shape

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    tiff_path_list = [pathlib.Path(n)
                      for n in z_stack_path_to_metadata.keys()]

    to_replace = 'mesoscope_file_splitter.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace, new=mock_read_metadata):
        with pytest.raises(RuntimeError,
                           match="has odd shape"):
            ZStackSplitter(tiff_path_list=tiff_path_list)


@pytest.fixture(scope='session')
def z_stack_roi_missing_tiff(tmp_path_factory,
                             baseline_zstack_metadata):
    """
    Create a set of z_stacks in which an ROI is not represented
    by discretePlaneMode==0 in the stack TIFFs
    """

    path_to_metadata = dict()
    tmp_dir = tmp_path_factory.mktemp('z_stack_many')
    for ii in range(3):
        tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.tiff')
        metadata = copy.deepcopy(baseline_zstack_metadata)
        rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
        if ii == 0 or ii == 1:
            rois[0]['discretePlaneMode'] = 0
        else:
            rois[ii]['discretePlaneMode'] = 0

        path_to_metadata[tmp_path] = metadata

    return path_to_metadata


def test_roi_missing_tiff(
        z_stack_roi_missing_tiff):
    """
    Test that an error is raised if an ROI does not have
    a corresponding z-stack TIFF with discretePlaneMode==0
    """
    z_stack_path_to_metadata = z_stack_roi_missing_tiff

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    tiff_path_list = [pathlib.Path(n)
                      for n in z_stack_path_to_metadata.keys()]

    to_replace = 'mesoscope_file_splitter.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace, new=mock_read_metadata):
        with pytest.raises(RuntimeError,
                           match="are represented in the local z-stack"):
            ZStackSplitter(tiff_path_list=tiff_path_list)


@pytest.fixture(scope='session')
def z_stack_bad_channelSave(tmp_path_factory,
                            baseline_zstack_metadata):
    """
    Create a set of z_stacks, one of which has an invalid
    value for channelSave
    """

    path_to_metadata = dict()
    tmp_dir = tmp_path_factory.mktemp('z_stack_many')
    for ii in range(3):
        tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.tiff')
        metadata = copy.deepcopy(baseline_zstack_metadata)
        if ii == 1:
            metadata[0]['SI.hChannels.channelSave'] = 1

        path_to_metadata[tmp_path] = metadata

    return path_to_metadata


def test_zstack_bad_channelSave(
        z_stack_bad_channelSave):
    """
    Test that an error is raised if a zstack file
    does not have channelSave==[1, 2]
    """
    z_stack_path_to_metadata = z_stack_bad_channelSave

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    tiff_path_list = [pathlib.Path(n)
                      for n in z_stack_path_to_metadata.keys()]

    to_replace = 'mesoscope_file_splitter.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace, new=mock_read_metadata):
        with pytest.raises(RuntimeError,
                           match="can only handle channelSave"):
            ZStackSplitter(tiff_path_list=tiff_path_list)
