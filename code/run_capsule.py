from typing import List, Tuple, Optional, Dict, Union
import tifffile
import numpy as np
import h5py as h5
import json
import os
import time
import logging
from pathlib import Path
import shutil
import tempfile
import datetime
import argparse
from pydantic import Field
from pydantic_settings import BaseSettings
import sys

from aind_ophys_utils.array_utils import normalize_array
from tiff_metadata import ScanImageMetadata
from full_field_utils import (
    write_out_stitched_full_field_image,
    get_full_field_path,
)


def mkstemp_clean(
    dir: Optional[Union[Path, str]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    A thin wrapper around tempfile mkstemp that automatically
    closes the file descripter returned by mkstemp.

    Parameters
    ----------
    dir: Optional[Union[Path, str]]
        The directory where the tempfile is created

    prefix: Optional[str]
        The prefix of the tempfile's name

    suffix: Optional[str]
        The suffix of the tempfile's name

    Returns
    -------
    file_path: str
        Path to a valid temporary file

    Notes
    -----
    Because this calls tempfile mkstemp, the file will be created,
    though it will be empty. This wrapper is needed because
    mkstemp automatically returns an open file descriptor, which was
    causing some of our unit tests to overwhelm the OS's limit
    on the number of open files.
    """
    (descriptor, file_path) = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)

    os.close(descriptor)
    return file_path


def split_timeseries_tiff(
    tiff_path: Path,
    offset_to_path: Dict,
    tmp_dir: Optional[Path] = None,
    dump_every: int = 1000,
    logger: Optional[callable] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Split a timeseries TIFF containing multiple mesoscope
    movies into individual HDF5 files.

    Parameters
    ----------
    tiff_path: Path
        Path to the timeseries TIFF file

    offset_to_path: Dict
        A dict mapping the offset corresponding to each timeseries
        to the path to the HDF5 file where that timeseries will be
        written (i.e. offset_to_path[0] = 'fileA.h5',
        offset_to_path[1] = 'fileB.h5', offset_to_path[2] = 'fileC.h5')
        corresponds to a TIFF file whose pages are arranged like
        [fileA, fileB, fileC, fileA, fileB, fileC, fileA, fileB, fileC....]

    tmp_dir: Optional[Path]
        Directory where temporary files are written (if None,
        the temporary files corresponding to each HDF5 file will
        be written to the directory where the final HDF5 is meant
        to be written)

    dump_every: int
        Write frames to the temporary files every dump_every
        frames per movie.

    logger: Optional[callable]
        Log statements will be written to logger.info()

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Returns
    -------
    None
        Timeseries are written to HDF5 files specified in offset_to_path

    Notes
    -----
    Because there is no way to get the number of pages in a BigTIFF
    without (expensively) scanning through all of the pages, this
    method operates by iterating through the pages once, writing
    the pages corresponding to each timeseries to a series of temporary
    files associated with that timeseries. Once all of the pages have
    been written to the temporary files, the temporary files associated
    with each timeseries are joined into the appropriate HDF5 files
    and deleted.
    """

    # Create a unique temporary directory with an unambiguous
    # name so that if clean-up gets interrupted by some
    # catastrophic failure we will know what the directory
    # was for.
    now = datetime.datetime.now()
    timestamp = f"{now.year}_{now.month}_" f"{now.day}_{now.hour}_" f"{now.minute}_{now.second}"

    tmp_prefix = f"mesoscope_timeseries_tmp_{timestamp}_"
    directories_to_clean = []

    if tmp_dir is not None:
        actual_tmp_dir = Path(tempfile.mkdtemp(dir=tmp_dir, prefix=tmp_prefix))
        directories_to_clean.append(actual_tmp_dir)

    offset_to_tmp_files = dict()
    offset_to_tmp_dir = dict()
    for offset in offset_to_path:
        if tmp_dir is not None:
            offset_to_tmp_dir[offset] = actual_tmp_dir
        else:
            pth = offset_to_path[offset]
            actual_tmp_dir = Path(tempfile.mkdtemp(dir=pth.parent, prefix=tmp_prefix))
            offset_to_tmp_dir[offset] = actual_tmp_dir
            directories_to_clean.append(actual_tmp_dir)
        offset_to_tmp_files[offset] = []

    try:
        _split_timeseries_tiff(
            tiff_path=tiff_path,
            offset_to_path=offset_to_path,
            offset_to_tmp_files=offset_to_tmp_files,
            offset_to_tmp_dir=offset_to_tmp_dir,
            dump_every=dump_every,
            logger=logger,
            metadata=metadata,
        )
    finally:
        for offset in offset_to_tmp_files:
            for tmp_pth in offset_to_tmp_files[offset]:
                if tmp_pth.exists():
                    os.unlink(tmp_pth)

        # clean up temporary directories
        for dir_pth in directories_to_clean:
            if dir_pth.exists():
                shutil.rmtree(dir_pth)


def _split_timeseries_tiff(
    tiff_path: Path,
    offset_to_path: Dict,
    offset_to_tmp_files: Dict,
    offset_to_tmp_dir: Dict,
    dump_every: int = 1000,
    logger: Optional[callable] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Method to do the work behind split_timeseries_tiff

    Parameters
    ----------
    tiff_path: Path
        Path to the timeseries TIFF being split

    offset_to_path: Dict
        Dict mapping offset to final HDF5 path (see
        split_timeseries_tiff for explanation)

    offset_to_tmp_files: Dict
        An empty dict for storing the lists of temporary
        files generated for each individual timeseries.
        This method wil populate the dict in place.

    offset_to_tmp_dir: Dict
        A dict mapping offset to the directory where
        the corresponding temporary files will be written

    dump_every: int
        Write to the temporary files every dump_every frames

    logger: Optional[callable]
        Log statements will be written to logger.info()

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Returns
    -------
    None
        Timeseries data is written to the paths specified in
        offset_to_path
    """

    if logger is not None:
        logger.info(f"Splitting {tiff_path}")

    max_offset = max(list(offset_to_path.keys()))

    fov_shape = None
    video_dtype = None
    offset_to_cache = dict()
    offset_to_valid_cache = dict()

    t0 = time.time()
    page_ct = 0
    with tifffile.TiffFile(tiff_path, mode="rb") as tiff_file:
        current_offset = -1
        cache_ct = 0
        for page in tiff_file.pages:
            page_ct += 1
            arr = page.asarray()

            if fov_shape is None:
                fov_shape = arr.shape
                video_dtype = arr.dtype
                for offset in offset_to_path:
                    cache = np.zeros((dump_every, fov_shape[0], fov_shape[1]), dtype=video_dtype)
                    offset_to_cache[offset] = cache

            current_offset += 1
            if current_offset > max_offset:
                current_offset = 0
                cache_ct += 1
                if cache_ct == dump_every:
                    _dump_timeseries_caches(
                        offset_to_cache=offset_to_cache,
                        offset_to_valid_cache=offset_to_valid_cache,
                        offset_to_tmp_files=offset_to_tmp_files,
                        offset_to_tmp_dir=offset_to_tmp_dir,
                    )
                    cache_ct = 0
                    if logger is not None:
                        duration = time.time() - t0
                        msg = f"Iterated through {page_ct} TIFF pages "
                        msg += f"in {duration:.2e} seconds"
                        logger.info(msg)

            offset_to_cache[current_offset][cache_ct, :, :] = arr
            offset_to_valid_cache[current_offset] = cache_ct + 1

    _dump_timeseries_caches(
        offset_to_cache=offset_to_cache,
        offset_to_valid_cache=offset_to_valid_cache,
        offset_to_tmp_files=offset_to_tmp_files,
        offset_to_tmp_dir=offset_to_tmp_dir,
    )

    if logger is not None:
        duration = time.time() - t0
        msg = f"Iterated through all {page_ct} TIFF pages "
        msg += f"in {duration:.2e} seconds"
        logger.info(msg)

    for offset in offset_to_tmp_files:
        _gather_timeseries_caches(
            file_path_list=offset_to_tmp_files[offset],
            final_output_path=offset_to_path[offset],
            metadata=metadata,
        )
        if logger is not None:
            duration = time.time() - t0
            msg = f"Wrote {offset_to_path[offset]} after "
            msg += f"{duration:.2e} seconds"
            logger.info(msg)

    if logger is not None:
        duration = time.time() - t0
        msg = f"Split {tiff_path} in {duration:.2e} seconds"
        logger.info(msg)


def _gather_timeseries_caches(
    file_path_list: List[Path], final_output_path: Path, metadata: Optional[dict] = None
) -> None:
    """
    Take a list of HDF5 files containing an array 'data' and
    join them into a single HDF5 file with an array 'data' that
    is the result of calling np.stack() on the smaller arrays.

    Parameters
    ----------
    file_path_list: List[Path]
        List of paths to files to be joined

    final_output_path: Path
        Path to the HDF5 file that is produced by joining
        file_path_list

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Return
    ------
    None
        Contents of files in file_path_list are joined into
        final_output_path

    Notes
    -----
    Files in file_path_list are deleted with os.unlink
    after they are joined.
    """
    n_frames = 0
    fov_shape = None
    video_dtype = None
    one_frame = None  # for calculating frame size in memory

    for file_path in file_path_list:
        with h5.File(file_path, "r") as in_file:
            n_frames += in_file["data"].shape[0]
            this_fov_shape = in_file["data"].shape[1:]

            if one_frame is None:
                one_frame = in_file["data"][0, :, :]

            if fov_shape is None:
                fov_shape = this_fov_shape
                video_dtype = in_file["data"].dtype
            else:
                if fov_shape != this_fov_shape:
                    raise RuntimeError("Inconsistent FOV shape\n" f"{fov_shape}\n{this_fov_shape}")

    # apparently, HDF5 chunks sizes must be less than 4 GB;
    # figure out how many frames fit in 3GB (just in case)
    # and set that as the maximum chunk size for the final
    # HDF5 file.
    three_gb = 3 * 1024**3
    bytes_per_frame = len(one_frame.tobytes())
    max_chunk_size = np.floor(three_gb / bytes_per_frame).astype(int)

    chunk_size = n_frames // 100

    if chunk_size < n_frames:
        chunk_size = n_frames

    if chunk_size > max_chunk_size:
        chunk_size = max_chunk_size

    with h5.File(final_output_path, "w") as out_file:
        if metadata is not None:
            serialized_metadata = json.dumps(metadata).encode("utf-8")
            out_file.create_dataset("scanimage_metadata", data=serialized_metadata)

        out_file.create_dataset(
            "data",
            shape=(n_frames, fov_shape[0], fov_shape[1]),
            dtype=video_dtype,
            chunks=(chunk_size, fov_shape[0], fov_shape[1]),
        )

        i0 = 0
        for file_path in file_path_list:
            with h5.File(file_path, "r") as in_file:
                chunk = in_file["data"][()]
                out_file["data"][i0 : i0 + chunk.shape[0], :, :] = chunk
                i0 += chunk.shape[0]
            os.unlink(file_path)


def _dump_timeseries_caches(
    offset_to_cache: Dict,
    offset_to_valid_cache: Dict,
    offset_to_tmp_files: Dict,
    offset_to_tmp_dir: Dict,
) -> None:
    """
    Write cached arrays to temporary files.

    Parameters
    ----------
    offset_to_cache: Dict
        Maps offset (see split_timeseries_tiff) to numpy
        arrays that are being dumped to temporary files.

    offset_to_valid_cache: Dict
        Maps offset to the index in offset_to_cache[offset] that
        is the last valid row (in case the cache was incompletely
        populated), i.e.
        offset_to_cache[offset][:offset_to_valid_cache[offset], :, :]
        is written to the temporary files.

        After this method is run, all entries in offset_to_cache
        are set to -1.

    offset_to_tmp_files: Dict
        Maps offset to list of temporary files that are being written.
        This dict starts out mapping to empty lists. This method
        creates temporary files and populates this dict with paths
        to them.

    offset_to_tmp_dir: Dict
       Maps offset to directory where temporary files will be written

    Returns
    -------
    None
        This metod writes the data input through offset_to_cache
        to temporary files (that this method creates and logs in
        offset_to_tmp_files)
    """

    for offset in offset_to_cache:
        tmp_dir = offset_to_tmp_dir[offset]
        valid = offset_to_valid_cache[offset]
        if valid < 0:
            continue
        cache = offset_to_cache[offset][:valid, :, :]

        tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".h5")

        tmp_path = Path(tmp_path)

        # append path first so the code knows to clean up
        # the file in case file-creation gets interrupted
        offset_to_tmp_files[offset].append(tmp_path)

        with h5.File(tmp_path, "w") as out_file:
            out_file.create_dataset("data", data=cache)
        offset_to_valid_cache[offset] = -1


class IntFromZMapperMixin(object):
    """
    This mixin defines methods to construct a mapping from
    floats to unique integer identifiers where floats are
    considered identical within some tolerance. The TIFF
    splitting classes use the methods provided to convert
    z-values which are floats into integers suitable for
    use as keys in dicts.
    """

    def _int_from_z(self, z_value: float, atol: float = 1.0e-6) -> int:
        """
        Convert a z_value into a unique integer, recording
        the mapping for reuse later, if necessary

        Parameters
        ----------
        z_value: float

        atol: float
           The absolute tolerance within which two floats
           are considered identical for the purposes of this
           method (if abs(f0-f1) <= atol, then it is considered
           that f0==f1)

           Note: if two recorded values are within atol of
           z_value, then the closest one is chosen.

        Returns
        -------
        int
           The unique integer associated with this z-value.
        """
        if not hasattr(self, "_int_from_z_lookup"):
            self._int_from_z_lookup = dict()
            self._z_from_int_lookup = dict()

        best_delta = None
        result = None
        max_value = -1
        for z_test, int_val in self._int_from_z_lookup.items():
            delta = np.abs(z_value - z_test)
            if int_val > max_value:
                max_value = int_val
            if delta <= atol:
                if best_delta is not None and delta > best_delta:
                    continue
                result = int_val
                best_delta = delta

        if result is None:
            new_val = max_value + 1
            self._int_from_z_lookup[z_value] = new_val
            result = new_val
            self._z_from_int_lookup[new_val] = z_value

        return result

    def _z_from_int(self, ii: int) -> float:
        """
        Return the float associated with a given integer
        in this lookup
        """
        return self._z_from_int_lookup[ii]


class TiffSplitterBase(IntFromZMapperMixin):
    """
    A class to naively split up a tiff file by just looping over
    the scanfields in its ROIs

    **this will not work for z-stacks**

    Parameters
    ----------
    tiff_path: Path
        Path to the TIFF file whose metadata we are parsing
    """

    def __init__(self, tiff_path: Path):
        self._file_path = tiff_path
        self._metadata = ScanImageMetadata(tiff_path=tiff_path)

        self._validate_z_stack()
        self._get_z_manifest()
        self._frame_shape = dict()

    @property
    def raw_metadata(self):
        """
        The ScanImage metadata as a dict
        """
        return self._metadata.raw_metadata

    def _validate_z_stack(self):
        """
        Make sure that the zsAllActuators are arranged the
        way we expect, i.e.
        [[roi0_z0, roi0_z1, roi0_z2..., roi0_zM],
         [roi0_zM+1, roi0_zM+2, ...],
         [roi1_z0, roi1_z1, roi1_z2..., roi1_zM],
         [roi1_zM+1, roi1_zM+2, ...],
         ...
         [roiN_z0, roiN_z1, roiN_z2...]]

        or, in the case of one ROI

        [z0, z1, z2....]
        """
        # check that self._metadata.channelSave is of a form
        # we can process
        if self._metadata.channelSave not in (1, [1, 2]):
            raise RuntimeError(
                "Expect channelSave == 1 or [1, 2]; got "
                f"{self._metadata.channelSave}\n{self._file_path}"
            )

        z_value_array = self._metadata.all_zs()

        # check that z_value_array is a list of lists
        if not isinstance(z_value_array, list):
            msg = "Unclear how to split this TIFF\n"
            msg += f"{self._file_path.resolve().absolute()}\n"
            msg += f"metadata.all_zs {self._metadata.all_zs()}"
            raise RuntimeError(msg)

        if isinstance(z_value_array[0], list):
            z_value_array = np.concatenate(z_value_array)

        # if self._metadata.channelSave == 1, verify that every
        # value of z_value_array is zero, then remove them
        if isinstance(self._metadata.channelSave, int):
            if self._metadata.channelSave != 1:
                raise RuntimeError(
                    "Expect channelSave == 1 or [1, 2]; got "
                    f"{self._metadata.channelSave}\n{self._file_path}"
                )
            for ii in range(1, len(z_value_array), 2):
                if z_value_array[ii] != 0:
                    raise RuntimeError(
                        "channelSave==1 but z values are "
                        f"{z_value_array}; "
                        "expect every other value to be zero\n"
                        f"{self._file_path}"
                    )
            z_value_array = z_value_array[::2]
        else:
            valid_channel = isinstance(self._metadata.channelSave, list)
            if valid_channel:
                valid_channel = self._metadata.channelSave == [1, 2]

            if not valid_channel:
                raise RuntimeError(
                    "Do not know how to handle channelSave=="
                    f"{self._metadata.channelSave}\n{self._file_path}"
                )

        defined_rois = self._metadata.defined_rois

        z_int_per_roi = []

        msg = ""

        # check that the same Z value does not appear more than
        # once in the same ROI
        for i_roi, roi in enumerate(defined_rois):
            if isinstance(roi["zs"], list):
                roi_zs = roi["zs"]
            else:
                roi_zs = [
                    roi["zs"],
                ]
            roi_z_ints = [self._int_from_z(z_value=zz) for zz in roi_zs]
            z_int_set = set(roi_z_ints)
            if len(z_int_set) != len(roi_zs):
                msg += f"roi {i_roi} has duplicate zs: {roi['zs']}\n"
            z_int_per_roi.append(z_int_set)

        # check that z values in z_array occurr in ROI order
        offset = 0
        n_roi = len(z_int_per_roi)
        n_z_per_roi = len(z_value_array) // n_roi

        # check that every ROI has the same number of zs
        if len(z_value_array) % n_roi != 0:
            msg += "There do not appear to be an "
            msg += "equal number of zs per ROI\n"
            msg += f"n_z: {len(z_value_array)} "
            msg += f"n_roi: {len(z_int_per_roi)}\n"

        for roi_z_ints in z_int_per_roi:
            these_z_ints = set(
                [
                    self._int_from_z(z_value=zz)
                    for zz in z_value_array[offset : offset + n_z_per_roi]
                ]
            )

            if these_z_ints != roi_z_ints:
                msg += "z_values from sub array "
                msg += "not in correct order for ROIs; "
                break
            offset += n_z_per_roi

        if len(msg) > 0:
            full_msg = "Unclear how to split this TIFF\n"
            full_msg += f"{self._file_path.resolve().absolute()}\n"
            full_msg += f"{msg}"
            full_msg += f"all_zs {self._metadata.all_zs()}\nfrom rois:\n"
            for roi in self._metadata.defined_rois:
                full_msg += f"zs: {roi['zs']}\n"
            raise RuntimeError(full_msg)

    def _get_z_manifest(self):
        """
        Populate various member objects that help us keep
        track of what z values go with what ROIs in this TIFF
        """
        local_z_value_list = np.array(self._metadata.all_zs()).flatten()
        defined_rois = self._metadata.defined_rois

        # create a list of sets indicating which z values were actually
        # scanned in the ROI (this will help us parse the placeholder
        # zeros that sometimes get dropped into
        # SI.hStackManager.zsAllActuators

        valid_z_int_per_roi = []
        valid_z_per_roi = []
        for roi in defined_rois:
            this_z_value = roi["zs"]
            if isinstance(this_z_value, int):
                this_z_value = [
                    this_z_value,
                ]
            z_as_int = [self._int_from_z(z_value=zz) for zz in this_z_value]
            valid_z_int_per_roi.append(set(z_as_int))
            valid_z_per_roi.append(this_z_value)

        self._valid_z_int_per_roi = valid_z_int_per_roi
        self._valid_z_per_roi = valid_z_per_roi
        self._n_valid_zs = 0
        self._roi_z_int_manifest = []
        ct = 0
        i_roi = 0
        local_z_index_list = [self._int_from_z(zz) for zz in local_z_value_list]
        for zz in local_z_index_list:
            if i_roi >= len(valid_z_int_per_roi):
                break
            if zz in valid_z_int_per_roi[i_roi]:
                roi_z = (i_roi, zz)
                self._roi_z_int_manifest.append(roi_z)
                self._n_valid_zs += 1
                ct += 1
                if ct == len(valid_z_int_per_roi[i_roi]):
                    i_roi += 1
                    ct = 0

    @property
    def input_path(self) -> Path:
        """
        The file this splitter is trying to split
        """
        return self._file_path

    def is_z_valid_for_roi(self, i_roi: int, z_value: float) -> bool:
        """
        Is specified z-value valid for the specified ROI
        """
        z_as_int = self._int_from_z(z_value=z_value)
        return z_as_int in self._valid_z_int_per_roi[i_roi]

    @property
    def roi_z_int_manifest(self) -> List[Tuple[int, int]]:
        """
        A list of tuples. Each tuple is a valid
        (roi_index, z_as_int) pair.
        """
        return self._roi_z_int_manifest

    @property
    def n_valid_zs(self) -> int:
        """
        The total number of valid z values associated with this TIFF.
        """
        return self._n_valid_zs

    @property
    def n_rois(self) -> int:
        """
        The number of ROIs in this TIFF
        """
        return self._metadata.n_rois

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        The (X, Y) center coordinates of roi_index=i_roi
        """
        return self._metadata.roi_center(i_roi=i_roi)

    def roi_size(self, i_roi: int) -> Tuple[float, float]:
        """
        The physical space size (x, y) of the i_roith ROI
        """
        return self._metadata.roi_size(i_roi=i_roi)

    def roi_resolution(self, i_roi: int) -> Tuple[int, int]:
        """
        The pixel resolution of the i_roith ROI
        """
        return self._metadata.roi_resolution(i_roi=i_roi)

    def _get_offset(self, i_roi: int, z_value: float) -> int:
        """
        Get the first page associated with the specified
        i_roi, z_value pair.
        """
        found_it = False
        n_step_over = 0
        this_roi_z = (i_roi, self._int_from_z(z_value=z_value))
        for roi_z_pair in self.roi_z_int_manifest:
            if roi_z_pair == this_roi_z:
                found_it = True
                break
            n_step_over += 1
        if not found_it:
            msg = f"Could not find stride for {i_roi}, {z_value}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)
        return n_step_over

    def frame_shape(self, i_roi: int, z_value: Optional[float]) -> Tuple[int, int]:
        """
        Get the shape of the image for a specified ROI at a specified
        z value

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: Optional[float]
            value of z. If None, z_value will be detected automaticall
            (assuming there is no ambiguity)

        Returns
        -------
        frame_shape: Tuple[int, int]
            (nrows, ncolumns)
        """
        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)

        key_pair = (i_roi, self._int_from_z(z_value))

        if key_pair not in self._frame_shape:
            offset = self._get_offset(i_roi=i_roi, z_value=z_value)
            with tifffile.TiffFile(self._file_path, mode="rb") as tiff_file:
                page = tiff_file.pages[offset].asarray()
                self._frame_shape[key_pair] = page.shape
        return self._frame_shape[key_pair]


class AvgImageTiffSplitter(TiffSplitterBase):
    @property
    def n_pages(self):
        """
        The number of pages in this TIFF
        """
        if not hasattr(self, "_n_pages"):
            with tifffile.TiffFile(self._file_path, mode="rb") as tiff_file:
                self._n_pages = len(tiff_file.pages)
        return self._n_pages

    def _get_pages(self, i_roi: int, z_value: float) -> List[np.ndarray]:
        """
        Get a list of np.ndarrays representing the pages of image data
        for ROI i_roi at the specified z_value
        """

        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} ROIs "
            msg += f"in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
            msg = f"{z_value} is not a valid z value for ROI {i_roi};"
            msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        offset = self._get_offset(i_roi=i_roi, z_value=z_value)

        with tifffile.TiffFile(self._file_path, mode="rb") as tiff_file:
            tiff_data = [
                tiff_file.pages[i_page].asarray()
                for i_page in range(offset, self.n_pages, self.n_valid_zs)
            ]

        key_pair = (i_roi, z_value)

        for arr in tiff_data:
            if key_pair in self._frame_shape:
                if arr.shape != self._frame_shape[key_pair]:
                    msg = f"ROI {i_roi} z_value {z_value}\n"
                    msg += "yields inconsistent frame shape"
                    raise RuntimeError(msg)
            else:
                self._frame_shape[key_pair] = arr.shape

        return tiff_data

    def _get_z_value(self, i_roi: int) -> float:
        """
        Return the z_value associated with i_roi, assuming
        there is only one. Raises a RuntimeError if there
        is more than one.
        """
        # When splitting surface TIFFs, there's no sensible
        # way to know the z-value ahead of time (whatever the
        # operator enters is just a placeholder). The block
        # of code below will scan for z-values than align with
        # the specified ROI ID and select the correct z value
        # (assuming there is only one)
        possible_z_values = []
        for pair in self.roi_z_int_manifest:
            if pair[0] == i_roi:
                possible_z_values.append(pair[1])
        if len(possible_z_values) > 1:
            msg = f"{len(possible_z_values)} possible z values "
            msg += f"for ROI {i_roi}; must specify one of\n"
            msg += f"{possible_z_values}"
            raise RuntimeError(msg)
        z_value = possible_z_values[0]
        return self._z_from_int(ii=z_value)

    def get_avg_img(self, i_roi: int, z_value: Optional[float]) -> np.ndarray:
        """
        Get the image created by averaging all of the TIFF
        pages associated with an (i_roi, z_value) pair

        Parameters
        ----------
        i_roi: int

        z_value: Optional[int]
            If None, will be detected automatically (assuming there
            is only one)

        Returns
        -------
        np.ndarray
            of floats
        """

        if not hasattr(self, "_avg_img_cache"):
            self._avg_img_cache = dict()

        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)

        z_int = self._int_from_z(z_value=z_value)
        pair = (i_roi, z_int)
        if pair not in self._avg_img_cache:
            data = np.array(self._get_pages(i_roi=i_roi, z_value=z_value))
            avg_img = np.mean(data, axis=0)
            self._avg_img_cache[pair] = avg_img

        return np.copy(self._avg_img_cache[pair])

    def write_output_file(self, i_roi: int, z_value: Optional[float], output_path: Path) -> None:
        """
        Write the image created by averaging all of the TIFF
        pages associated with an (i_roi, z_value) pair to a TIFF
        file.

        Parameters
        ----------
        i_roi: int

        z_value: Optional[int]
            If None, will be detected automatically (assuming there
            is only one)

        output_path: Path
            Path to file to be written

        Returns
        -------
        None
            Output is written to output_path
        """

        if output_path.suffix not in (".tif", ".tiff"):
            msg = "expected .tiff output path; "
            msg += f"you specified {output_path.resolve().absolute()}"

        avg_img = self.get_avg_img(i_roi=i_roi, z_value=z_value)

        avg_img = normalize_array(array=avg_img, lower_cutoff=None, upper_cutoff=None)

        metadata = {"scanimage_metadata": self.raw_metadata}

        tifffile.imwrite(output_path, avg_img, metadata=metadata)
        return None


class TimeSeriesSplitter(TiffSplitterBase):
    """
    A class specifically for splitting timeseries TIFFs

    Parameters
    ----------
    tiff_path: Path
        Path to the TIFF file whose metadata we are parsing
    """

    def write_output_files(
        self,
        output_path_map: Dict[Tuple[int, float], Path],
        tmp_dir: Optional[Path] = None,
        dump_every: int = 1000,
        logger: Optional[callable] = None,
    ) -> None:
        """
        Write all of the pages associated with an
        (i_roi, z_value) pair to an HDF5 file.

        Parameters
        ----------
        output_path_map: Dict[Tuple[int, float], Path]
            Dict mapping (i_roi, z_value) pairs to output paths
            where the data for those ROIs should go.

        tmp_dir: Optional[Path]
            Directory where temporary files will be written.

        dump_every: int
            Number of frames to store in each temprorary file
            (see Notes)

        logger: Optional[callable]
            Logger which will be invoked with logger.INFO
            by worker methods

        Returns
        -------
        None
            Timeseries for the ROIs are written to the paths
            specified in output_path_map.

            If not specified, will write temporary files into
            the directory where the final files are meant to
            be written.

        Notes
        -----
        Because the only way to get n_pages from a BigTIFF is to
        iterate over its pages, counting, this module works by iterating
        over the pages, splitting the timeseries data into small temp files
        as it goes and keeping track of how many total pages are being written
        for each ROI. After the temp files are written, the temp files are
        gathered together into the final files specified in output_path_map
        and the temporary files are deleted.
        """
        for key_pair in output_path_map:
            output_path = output_path_map[key_pair]
            if output_path.suffix != ".h5":
                msg = "expected HDF5 output path; "
                msg += f"you gave {output_path.resolve().absolute()}"
                raise ValueError(msg)

            i_roi = key_pair[0]
            z_value = key_pair[1]

            if i_roi < 0:
                msg = f"You asked for ROI {i_roi}; "
                msg += "i_roi must be >= 0"
                raise ValueError(msg)

            if i_roi >= self.n_rois:
                msg = f"You asked for ROI {i_roi}; "
                msg += f"there are only {self.n_rois} ROIs "
                msg += f"in {self._file_path.resolve().absolute()}"
                raise ValueError(msg)

            if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
                msg = f"{z_value} is not a valid z value for ROI {i_roi};"
                msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
                msg += f"TIFF file {self._file_path.resolve().absolute()}"
                raise ValueError(msg)

        if len(output_path_map) != len(self.roi_z_int_manifest):
            msg = f"you specified paths for {len(output_path_map)} "
            msg += "timeseries files, but the metadata for this "
            msg += "TIFF says it contains "
            msg += f"{len(self.roi_z_int_manifest)}; "
            msg += "we cannot split this file "
            msg += f"({self._file_path})"
            raise ValueError(msg)

        all_roi_z_int = set()
        for key_pair in output_path_map:
            i_roi = key_pair[0]
            z_value = key_pair[1]
            this_pair = (i_roi, self._int_from_z(z_value=z_value))
            all_roi_z_int.add(this_pair)
        for roi_z_pair in self.roi_z_int_manifest:
            if roi_z_pair not in all_roi_z_int:
                raise ValueError(
                    "You did not specify output paths for all "
                    "of the timeseries in "
                    f"{self._file_path}"
                )

        offset_to_path = dict()
        for key_pair in output_path_map:
            i_roi = key_pair[0]
            z_value = key_pair[1]
            offset = self._get_offset(i_roi=i_roi, z_value=z_value)
            if offset in offset_to_path:
                raise RuntimeError("Same offset occurs twice when splitting " f"{self._file_path}")
            offset_to_path[offset] = output_path_map[key_pair]

        split_timeseries_tiff(
            tiff_path=self._file_path,
            tmp_dir=tmp_dir,
            offset_to_path=offset_to_path,
            dump_every=dump_every,
            logger=logger,
            metadata=self.raw_metadata,
        )

        return None


class ZStackSplitter(IntFromZMapperMixin):
    """
    Class to handle splitting all of the _local_z_stack.tiff files
    associated with an OPhys session.

    Parameters
    ----------
    tiff_path_list: List[Path]
        List of paths to the _local_z_stack.tiff files
        associated with this OPhys session.
    """

    def __init__(self, tiff_path_list: List[Path]):
        self._path_to_metadata = dict()
        self._frame_shape = dict()
        for tiff_path in tiff_path_list:
            str_path = str(tiff_path.resolve().absolute())
            self._path_to_metadata[str_path] = ScanImageMetadata(tiff_path=tiff_path)

            if self._path_to_metadata[str_path].channelSave != [1, 2]:
                raise RuntimeError(
                    f"metadata for {str_path} has channelSave="
                    f"{self._path_to_metadata[str_path].channelSave}\n"
                    "can only handle channelSave==[1, 2]"
                )

        # construct lookup tables to help us map ROI index and z-value
        # to a tiff path and an index in the z-array

        # map (i_roi, z_value) pairs to TIFF file paths
        self._roi_z_int_to_path = dict()

        # map (tiff_file_path, z_value) to the index, i.e.
        # to which scanned z-value *in this TIFF* does the
        # z-value correspond.
        self._path_z_int_to_index = dict()

        # this is an internal lookup table which we will use
        # to validate that every ROI is represented by a
        # z-stack file
        roi_to_path = dict()

        for tiff_path in self._path_to_metadata.keys():
            metadata = self._path_to_metadata[tiff_path]
            this_roi = None
            for i_roi, roi in enumerate(metadata.defined_rois):
                if roi["discretePlaneMode"] == 0:
                    if this_roi is not None:
                        raise RuntimeError(
                            "More than one ROI has " "discretePlaneMode==0 for " "{tiff_path}"
                        )
                    this_roi = i_roi

            if this_roi is None:
                raise RuntimeError("Could not find discretePlaneMode==0 for " f"{tiff_path}")

            if this_roi not in roi_to_path:
                roi_to_path[this_roi] = []
            roi_to_path[this_roi].append(tiff_path)

            z_array = np.array(metadata.all_zs())
            if z_array.shape[1] != 2:
                raise RuntimeError(f"z_array for {tiff_path} has odd shape\n" f"{z_array}")

            z_mean = z_array.mean(axis=0)
            for ii, z_value in enumerate(z_mean):
                roi_z = (this_roi, self._int_from_z(z_value=z_value))
                self._roi_z_int_to_path[roi_z] = tiff_path
                path_z = (tiff_path, self._int_from_z(z_value=z_value))
                self._path_z_int_to_index[path_z] = ii

        # check that every ROI has a z-stack file
        for tiff_path in self._path_to_metadata:
            metadata = self._path_to_metadata[tiff_path]
            n_rois = len(metadata.defined_rois)
            if len(roi_to_path) != n_rois:
                msg = (
                    f"There are {n_rois} ROIs; however, only "
                    f"{len(roi_to_path)} of them are represented in the "
                    "local z-stack TIFFS. Here is a mapping from i_roi to "
                    "TIFF paths\n"
                    f"{json.dumps(roi_to_path, indent=2, sort_keys=True)}"
                    "\n\nThis was determined by scanning the z-stack TIFFs "
                    "and noting which ROIs were marked with "
                    "discretePlaneMode==0"
                )
                raise RuntimeError(msg)

        self._path_to_pages = dict()
        for tiff_path in self._path_to_metadata.keys():
            with tifffile.TiffFile(tiff_path, mode="rb") as tiff_file:
                self._path_to_pages[tiff_path] = len(tiff_file.pages)

    @property
    def input_path(self) -> List[str]:
        """
        The list of files this splitter is trying to split
        """
        output = list(self._path_to_metadata.keys())
        output.sort()
        return output

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        Return the (X, Y) center coordinates for the ROI specified
        by i_roi.
        """
        center_tol = 1.0e-5
        possible_center = []
        for pair in self._roi_z_int_to_path:
            if pair[0] != i_roi:
                continue
            tiff_path = self._roi_z_int_to_path[pair]
            metadata = self._path_to_metadata[tiff_path]
            possible_center.append(metadata.roi_center(i_roi=i_roi))

        baseline_center = possible_center[0]
        for ii in range(1, len(possible_center)):
            center = possible_center[ii]
            dsq = (center[0] - baseline_center[0]) ** 2 + (center[1] - baseline_center[1]) ** 2
            if dsq > center_tol:
                msg = "Cannot find consistent center for ROI "
                msg += f"{i_roi}"
        return baseline_center

    def frame_shape(self, i_roi: int, z_value: float) -> Tuple[int, int]:
        """
        Return the (nrows, ncolumns) shape of the first page associated
        with the specified (i_roi, z_value) pair
        """
        roi_z = (i_roi, self._int_from_z(z_value=z_value))
        tiff_path = self._roi_z_int_to_path[roi_z]

        path_z = (tiff_path, self._int_from_z(z_value=z_value))
        z_index = self._path_z_int_to_index[path_z]

        with tifffile.TiffFile(tiff_path, mode="rb") as tiff_file:
            page = tiff_file.pages[z_index].asarray()
        return page.shape

    def _get_tiff_path(self, i_roi: int, z_value: float) -> Path:
        """
        Return the tiff path corresponding to the (i_roi, z_value) pair
        """
        roi_z = (i_roi, self._int_from_z(z_value=z_value))
        tiff_path = self._roi_z_int_to_path[roi_z]
        return tiff_path

    def _get_pages(self, i_roi: int, z_value: float) -> np.ndarray:
        """
        Get all of the TIFF pages associated in this z-stack set with
        an (i_roi, z_value) pair. Return as a numpy array shaped like
        (n_pages, nrows, ncolumns)
        """
        tiff_path = self._get_tiff_path(i_roi=i_roi, z_value=z_value)

        path_z = (tiff_path, self._int_from_z(z_value=z_value))
        z_index = self._path_z_int_to_index[path_z]

        data = []
        n_pages = self._path_to_pages[tiff_path]
        baseline_shape = self.frame_shape(i_roi=i_roi, z_value=z_value)
        with tifffile.TiffFile(tiff_path, mode="rb") as tiff_file:
            data = [tiff_file.pages[i_page].asarray() for i_page in range(z_index, n_pages, 2)]

        for this_page in data:
            if this_page.shape != baseline_shape:
                msg = f"ROI {i_roi} z_value {z_value} "
                msg += "give inconsistent page shape"
                raise RuntimeError(msg)

        return np.stack(data)

    def write_output_file(self, i_roi: int, z_value: float, output_path: Path) -> None:
        """
        Write the z-stack for a specific ROI, z pair to an
        HDF5 file

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: int
            depth of the plane whose z-stack we are writing

        output_path: Path
            path to the HDF5 file to be written

        Returns
        -------
        None
            output is written to the HDF5 file.
        """

        if output_path.suffix != ".h5":
            msg = "expected HDF5 output path; "
            msg += f"you gave {output_path.resolve().absolute()}"
            raise ValueError(msg)

        data = self._get_pages(i_roi=i_roi, z_value=z_value)

        metadata = self._path_to_metadata[
            self._get_tiff_path(i_roi=i_roi, z_value=z_value)
        ].raw_metadata

        with h5.File(output_path, "w") as out_file:
            out_file.create_dataset("scanimage_metadata", data=json.dumps(metadata).encode("utf-8"))

            out_file.create_dataset("data", data=data, chunks=(1, data.shape[1], data.shape[2]))


def get_valid_roi_centers(timeseries_splitter: TimeSeriesSplitter) -> List[Tuple[float, float]]:
    """
    Return a list of all of the valid ROI centers taken from a
    TimeSeriesSplitter
    """
    eps = 0.01  # ROIs farther apart than this are different
    valid_roi_centers = []
    for roi_z_tuple in timeseries_splitter.roi_z_int_manifest:
        roi_center = timeseries_splitter.roi_center(i_roi=roi_z_tuple[0])
        dmin = None
        for other_roi_center in valid_roi_centers:
            distance = np.sqrt(
                (roi_center[0] - other_roi_center[0]) ** 2
                + (roi_center[1] - other_roi_center[1]) ** 2
            )
            if dmin is None or distance < dmin:
                dmin = distance
        if dmin is None or dmin > eps:
            valid_roi_centers.append(roi_center)
    return valid_roi_centers


def get_nearest_roi_center(
    this_roi_center: Tuple[float, float], valid_roi_centers: List[Tuple[float, float]]
) -> int:
    """
    Take a specified ROI center and a list of valid ROI centers,
    return the index in valid_roi_centers that is closest to
    this_roi_center
    """
    dmin = None
    ans = None
    for i_roi, roi in enumerate(valid_roi_centers):
        dist = (this_roi_center[0] - roi[0]) ** 2 + (this_roi_center[1] - roi[1]) ** 2
        if dmin is None or dist < dmin:
            ans = i_roi
            dmin = dist

    if ans is None:
        msg = "Could not find nearest ROI center for\n"
        msg += f"{this_roi_center}\n"
        msg += f"{valid_roi_centers}\n"
        raise RuntimeError(msg)

    return ans

class JobSettings(BaseSettings):
    """Job settings values."""

    storage_path: Union[Path, str] = Field(description="directory where tiff files are found")
    temp_dir: Optional[Path]

class TiffSplitterCLI:
    def __init__(self, job_settings=JobSettings):
        self.storage_path = job_settings.storage_path
        
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)
        session_json = next(self.storage_path.glob("*session.json"), None)
        if not session_json:
            raise ValueError("No session.json file found")
        with open(session_json, "r") as f:
            self.session_data = json.load(f)
        self.timeseries_tif = next(self.storage_path.glob("*timeseries.tiff"), None)
        if not self.timeseries_tif:
            raise ValueError("No timeseries.tiff file found")
        self.depths_tif = next(self.storage_path.glob("*averaged_depth.tiff"), None)
        if not self.depths_tif:
            raise ValueError("No averaged_depth.tiff file found")
        self.surface_tif = next(self.storage_path.glob("*averaged_surface.tiff"), None)
        if not self.surface_tif:
            raise ValueError("No averaged_surface.tiff file found")
        self.temp_dir = job_settings.temp_dir

    def run_job(self):
        t0 = time.time()
        output = {"column_stacks": []}
        files_to_record = []

        ts_path = Path(self.timeseries_tif)
        timeseries_splitter = TimeSeriesSplitter(tiff_path=ts_path)
        files_to_record.append(ts_path)

        depth_path = Path(self.depths_tif)
        depth_splitter = AvgImageTiffSplitter(tiff_path=depth_path)
        files_to_record.append(depth_path)

        surface_path = Path(self.surface_tif)
        surface_splitter = AvgImageTiffSplitter(tiff_path=surface_path)
        files_to_record.append(surface_path)

        zstack_path_list = []
        for zstack in self.storage_path.glob("*local_z*.tiff"):
            zstack_path = zstack
            zstack_path_list.append(zstack_path)
            files_to_record.append(zstack_path)

        zstack_splitter = ZStackSplitter(tiff_path_list=zstack_path_list)

        ready_to_archive = set()

        # Looking at recent examples of outputs from this queue,
        # I do not think we have honored the 'column_z_stack_tif'
        # entry in the schema for some time now. I find no examples
        # in which this entry of the input.jon is ever populated.
        # I am leaving it here for now to avoid the complication of
        # having to modify the ruby strategy associated with this
        # queue, which is out of scope for the work we have
        # currently committed to.
        if self.storage_path.glob("*cortical_z_stack*.tiff"):
            msg = "'column_z_stack_tif' detected in 'plane_groups'; "
            msg += "the TIFF splitting code no longer handles that file."
            logging.warning(msg)

        # There are cases where the centers for ROIs are not
        # exact across modalities, so we cannot demand that the
        # ROI centers be the same to within an absolute tolerance.
        # Here we use the timeseries TIFF to assemble a list of all
        # available ROI centers. When splitting the other TIFFs, we
        # will validate them by making sure that the closest
        # valid_roi_center is always what we expect.
        valid_roi_centers = get_valid_roi_centers(timeseries_splitter=timeseries_splitter)

        experiment_metadata = []
        for data_stream in self.session_data["data_streams"]:
            if data_stream.get("ophys_fovs"):
                for fov in data_stream["ophys_fovs"]:
                    this_exp_metadata = dict()
                    fov_id = f'{fov["targeted_structure"]}_{fov["index"]}'
                    this_exp_metadata["fov"] = fov_id
                    for file_key in ("timeseries", "depth_2p", "surface_2p", "local_z_stack"):
                        this_metadata = dict()
                        for data_key in (
                            "fov_coordinate_ml",
                            "fov_coordinate_ap",
                            "fov_reference",
                            "scanimage_resolution",
                        ):
                            this_metadata[data_key] = fov[data_key]
                        this_exp_metadata[file_key] = this_metadata

                    fov_directory = self.storage_path
                    roi_index = fov["scanimage_roi_index"]
                    scanfield_z = fov["scanfield_z"]
                    baseline_center = None

                    for splitter, z_value, output_name, metadata_tag in zip(
                        (depth_splitter, surface_splitter, zstack_splitter),
                        (scanfield_z, None, scanfield_z),
                        (
                            f"{fov_id}_depth.tif",
                            f"{fov_id}_surface.tif",
                            f"{fov_id}_z_stack_local.h5",
                        ),
                        ("depth_2p", "surface_2p", "local_z_stack"),
                    ):
                        output_dir = fov_directory / fov_id
                        if not output_dir.is_dir():
                            output_dir.mkdir()
                        output_path = output_dir / output_name
                        roi_center = splitter.roi_center(i_roi=roi_index)
                        nearest_valid = get_nearest_roi_center(
                            this_roi_center=roi_center, valid_roi_centers=valid_roi_centers
                        )
                        if baseline_center is None:
                            baseline_center = nearest_valid

                        if nearest_valid != baseline_center:
                            msg = f"experiment {fov_id}\n"
                            msg += "roi center inconsistent for "
                            msg += "input: "
                            msg += f"{splitter.input_path.resolve().absolute()}\n"
                            msg += "output: "
                            msg += f"{output_path.resolve().absolute()}\n"
                            msg += f"{baseline_center}; {roi_center}\n"
                            raise RuntimeError(msg)
                        splitter.write_output_file(
                            i_roi=roi_index, z_value=z_value, output_path=output_path
                        )
                        str_path = str(output_path.resolve().absolute())
                        this_exp_metadata[metadata_tag]["filename"] = str_path
                        frame_shape = splitter.frame_shape(i_roi=roi_index, z_value=z_value)
                        this_exp_metadata[metadata_tag]["height"] = frame_shape[0]
                        this_exp_metadata[metadata_tag]["width"] = frame_shape[1]

                        elapsed_time = time.time() - t0

                        logging.info(
                            "wrote "
                            f"{output_path.resolve().absolute()} "
                            f"after {elapsed_time:.2e} seconds"
                        )

                    experiment_metadata.append(this_exp_metadata)

        # because the timeseries TIFFs are so big, we split
        # them differently to avoid reading through the TIFFs
        # more than once

        output_path_lookup = dict()
        for data_stream in self.session_data["data_streams"]:
            if data_stream.get("ophys_fovs"):
                for fov in data_stream["ophys_fovs"]:
                    fov_id = f'{fov["targeted_structure"]}_{fov["index"]}'
                    scanfield_z = fov["scanfield_z"]
                    roi_index = fov["scanimage_roi_index"]
                    fov_directory = self.storage_path
                    fname = f"{fov_id}.h5"
                    output_path = fov_directory / fov_id / fname
                    output_path_lookup[(roi_index, scanfield_z)] = output_path

                    frame_shape = timeseries_splitter.frame_shape(i_roi=roi_index, z_value=scanfield_z)

                    str_path = str(output_path.resolve().absolute())

                    for metadata in experiment_metadata:
                        if metadata["fov"] == fov:
                            metadata["timeseries"]["height"] = frame_shape[0]
                            metadata["timeseries"]["width"] = frame_shape[1]
                            metadata["timeseries"]["filename"] = str_path
        timeseries_splitter.write_output_files(
            output_path_map=output_path_lookup,
            tmp_dir=self.temp_dir,
            dump_every=self.session_data.get("dump_every", 1000),
            logger=logging,
        )

        output["experiment_output"] = experiment_metadata

        ready_to_archive.add(self.surface_tif)
        ready_to_archive.add(self.depths_tif)
        ready_to_archive.add(self.timeseries_tif)
        for zstack_path in zstack_path_list:
            ready_to_archive.add(str(zstack_path.resolve().absolute()))

        output["ready_to_archive"] = list(ready_to_archive)

        # TODO: why not just search for the file in the storage_path?
        # full_field_path = get_full_field_path(runner_args=self.args, logger=logging)
        full_field_path = self.storage_path.glob("*fullfield*.tiff")

        if full_field_path is not None:
            avg_path = self.surface_tif
            output_dir = self.storage_path

            output_name = "stitched_full_field_img.h5"
            output_path = output_dir / output_name
            logging.info(f"Writing {output_path.resolve().absolute()}")

            write_out_stitched_full_field_image(
                path_to_avg_tiff=Path(avg_path),
                path_to_full_field_tiff=full_field_path,
                output_path=output_path
            )

            if output_path.is_file():
                logging.info(
                    "Wrote full field stitched image to " f"{output_path.resolve().absolute()}"
                )

        # record file metadata
        file_metadata = []
        for file_path in files_to_record:
            tiff_metadata = ScanImageMetadata(file_path)
            this_metadata = dict()
            this_metadata["input_tif"] = str(file_path.resolve().absolute())
            this_metadata["scanimage_metadata"] = tiff_metadata._metadata[0]
            this_metadata["roi_metadata"] = tiff_metadata._metadata[1]
            file_metadata.append(this_metadata)
        output["file_metadata"] = file_metadata

        duration = time.time() - t0
        logging.info(f"that took {duration:.2e} seconds")

    @classmethod
    def from_args(cls, args: list):
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-u",
            "--job-settings",
            required=True,
            type=json.loads,
            help=(
                r"""
                Custom settings defined by the user defined as a json
                 string. For example: -u
                 '{"storage_path":"../data",
                 "tmp_dir":"../scratch""}
                """
            ),
        )
        job_args = parser.parse_args(args)
        job_settings_from_args = JobSettings(**job_args.job_settings)
        return cls(
            job_settings=job_settings_from_args,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    runner = TiffSplitterCLI.from_args(sys_args)
    runner.run_job()
