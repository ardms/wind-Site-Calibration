# Site Calibration Based on IEC 61400-12-1

This repository contains a comprehensive tool for performing site calibration
consistent with the IEC 61400-12-1 wind turbine power performance testing
standard. It includes specific filtering criteria and methodologies for
assessing the significance of wind shear, supporting both methods outlined in
the standard for site calibration.

## Introduction

Site calibration is critical to ensuring that wind measurements from anemometry
are reliable and an accurate representation of the wind profile as per IEC
standards. This tool incorporates advanced filtering based on both IEC
specifications and additional criteria recommended by industry experts such as
Vestas. It allows users to perform calibration using two different methods,
with the choice of method depending on the statistical significance of wind
shear in the collected dataset.

## Prerequisites

Please ensure that you have the following prerequisites before commencing:

* Pandas and Rich Python libraries
* Wind speed measurements from two Met mast one would be the __Control__ (CMM) and one would be the __Permanent__ (PMM)

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/ardms/wind-Site-Calibration
cd site-calibration-iec61400
# If using Python, for example:
pip install -r requirements.txt
```

# Usage

The code base requires a configuration file to define all necessary parameters.
A [template](./config-template.json) is provided for that reason

## Data Filtering
Filter raw measurement data using these IEC specified criteria:

1. NumSamplesInterval_avg_PMM2 == 600
2. DirNumSamples > 595
3. Temperature > 2°C and Humidity < 80%
4. Mean wind speed within range 4 to 16 m/s
5. Measurement sector: Defined by the wake effects of the selected Turbine

Additional filtering to be discussed on per case and as needed:

1. Inflow_angle_PMM within ±2°
2. TI_V1_PMM between 6% and 24%
3. Alpha_V1_V3_PMM between 0 and 0.5
4. Precipitation_avg_PMM2 < 10mm

## Direction
As detailed in the IEC, the wind direction bin size shall be 10°.

## Significance of shear
