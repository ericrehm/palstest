"""Reusable I/O for ac-s IOP downcast profiles and ship navigation data.

acs: load one acsPROdowncast .txt profile into a DataFrame with a merged
     ISO-8601 UTC datetime column.
nav: load ship nav data (outgoing_data/*.csv, 1-minute samples) into a
     single time-sorted DataFrame spanning the whole cruise.
profiles: match lidar shots to contemporaneous ac-s casts by time, and
     depth-average c_water-corrected total attenuation ct(532)/ct(650)
     over a fit window -- used by v2's L2 processing to add
     ct532_near/ct532_far/ct650_near/ct650_far.
"""
from .acs import load_wavelengths, load_acs_cast
from .nav import load_nav_file, load_nav_directory, clean_lat_lon, DATETIME_COL, LAT_COL, LON_COL
from .profiles import IOPProfileSet, IOPWindowAverage, OUTPUT_SPECS, MATCH_PAD

__all__ = [
    "load_wavelengths",
    "load_acs_cast",
    "load_nav_file",
    "load_nav_directory",
    "clean_lat_lon",
    "DATETIME_COL",
    "LAT_COL",
    "LON_COL",
    "IOPProfileSet",
    "IOPWindowAverage",
    "OUTPUT_SPECS",
    "MATCH_PAD",
]
