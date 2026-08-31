"""
QA/QC flag vocabulary shared across every quality test in the PALS pipeline
(ADC occupancy here in L1; more will follow in L1/L2/L3).

Flag codes follow IOOS QARTOD (Quality Assurance / Quality Control of
Real-Time Oceanographic Data) conventions:
https://cdn.ioos.noaa.gov/media/2017/12/QARTOD-Data-Flags-Manual_Final.pdf

    1 = GOOD     Pass. Passed critical real-time QC tests; ready for general use.
    2 = UNKNOWN  Not Evaluated. Not tested, or quality info unavailable.
    3 = SUSPECT  Suspect / Of High Interest. Questionable; needs human review.
    4 = FAIL     Fail / Bad. Failed critical QC checks; not acceptable for standard use.
    9 = MISSING  Data absent, or used as a placeholder.

Not every QA test emits all five codes -- a test uses whichever subset
applies to it. E.g. ADC occupancy (see l1_processor._adc_occupancy) never
emits UNKNOWN in normal operation, since Q_ADC is always computable from data
L1 always has; MISSING is still handled defensively for NaN/absent bins.
"""

from enum import IntEnum


class QartodFlag(IntEnum):
    GOOD = 1
    UNKNOWN = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9
