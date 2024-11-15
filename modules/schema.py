from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from pathlib import Path


class MetMast(BaseModel):
    id: int = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    name: str
    date_commission: datetime | None = None
    anemo_names: list | None = None
    anemo_heights: list | None = None
    vanes_names: list | None = None
    vanes_heights: list | None = None
    thermo_names: list | None = None
    thermo_heights: list | None = None
    vertical_anemo_names: list | None = None
    vertical_anemo_heights: list | None = None
    presipitation_names: list | None = None
    presipitation_heights: list | None = None
    date_type: str | None = None
    timeseries_path: Path | None = None
    timeseries_skiprows: list | None = None
    timeseries_header: int | None = None
    timeseries_index_col: str | None = None
    timeseries: list | None = None

    def load_timeseries(self):
        """
        Read raw data files and save them in the class
        """
        pass
        df = pd.read_csv(
            self.timeseries_path.joinpath("VC23254_Stranoch_PMMS2_DATA_10MIN.dat"),
            header=self.timeseries_header,
            skiprows=self.timeseries_skiprows,
            index_col=self.timeseries_index_col,
        )
        self.timeseries = df

    def load_timeseries_folder(self):
        """
        Read all files in a folder and merge them in one time series. Save this timeseries in the class
        """
        pass

    def IEC_filtering(self):
        """
        Filtering met mast dataset based on IEC 61400-12-1 spesifications for site calibrations
        1 (NumSamplesInterval_avg_PMM2 == 600)
        2 (NumSamplesInterval_avg_CMM2 == 600)
        3 (V1_avg_CMM2_cor >= 4 & V1_avg_CMM2_cor <= 16)
        4 (Dir1_avg_PMM2 >= 206.6 | Dir1_avg_PMM2 <= 6.9)
        5 (inflow_angle_PMM >= -2 & inflow_angle_PMM <= 2)
        6 (TI_V1_PMM >= 0.06 & TI_V1_PMM <= 0.24)
        7 (alpha_V1_V3_PMM >= 0 & alpha_V1_V3_PMM <= 0.5)
        8 (Precipitation_avg_PMM2 < 10)
        """
        pass


class Site(BaseModel):
    id: int = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    name: str
    filter_NumSamplesInterval: int | None = None
    filter_WindAvgMin: int | None = None
    filter_WindAvgMax: int | None = None
    filter_DirAvgMin: int | None = None
    filter_DirAvgMax: int | None = None
    filter_InflowAngleMin: int | None = None
    filter_InflowAngleMax: int | None = None
    filter_TurbulanceIntencityMin: int | None = None
    filter_TurbulanceIntencityMax: int | None = None
    filter_AlphaMin: int | None = None
    filter_AlphaMax: int | None = None
    filter_PrecipitationMin: int | None = None
    filter_PrecipitationMax: int | None = None
    CMM: MetMast
    PMM: MetMast

    def site_calibration(self):
        """
        This is the main method that will perform site calibrations
        """
        pass
