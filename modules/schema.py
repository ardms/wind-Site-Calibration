import re
import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")


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
    data_type: str | None = None
    timeseries_path: Path | None = None
    timeseries_skiprows: list | None = None
    timeseries_header: int | None = None
    timeseries_index_col: str | None = None
    timeseries: list | None = None

    def load_timeseries_file(self, file):
        """
        Read raw data files and save them in the class
        """
        df = pd.read_csv(
            file,
            header=self.timeseries_header,
            skiprows=self.timeseries_skiprows,
            index_col=self.timeseries_index_col,
            low_memory=False,
        )
        # self.timeseries = df
        return df.astype(np.float64)

    def load_timeseries_from_folder(self):
        """
        Read all files in a folder and merge them in one time series. Save this timeseries in the class
        """
        folder = Path(self.timeseries_path)
        list_of_dfs = []
        for file in folder.glob(f"*.{self.data_type}"):
            log.info(f"Working on {file.name}")
            df = self.load_timeseries_file(file)
            list_of_dfs.append(df)
        try:
            self.timeseries = pd.concat(list_of_dfs)
            log.info(f"All data from {folder} have been succesfully loaded")
        except Exception as ex:
            log.error(f"{ex}")

    def calculate_alpha(self, low_wind: (str, int), high_wind: (str, int)):
        """
        Calculate alpha (wind shear exponent) value of the wind shear
        based on the formula Vzi = Vh*(Zi/H)^a
        Return: None but will overwrite self.timeseries
        """
        df = self.timeseries
        numerator = np.where(
            df[high_wind[0]] > 0, np.log(df[low_wind[0]] / df[high_wind[0]]), np.nan
        )
        df["alpha"] = numerator / np.log(low_wind[1] / high_wind[1])
        log.info(
            f"Wind shear exponent has been calculated using {low_wind} and {high_wind}"
        )
        self.timeseries = df

    def calculate_TI(self):
        """
        Calculate Turbulence Intesity as a ratio of avg value devided by the standard deviation
        """
        df = self.timeseries
        columns = [i for i in df.columns if re.match("v._Avg", i)]
        for column in columns:
            stem = column.split("_")[0]
            std = f"{stem}_Std"
            avg = f"{stem}_Avg"
            ti = f"{stem}_TI"
            df[ti] = df[avg] / df[std]
        log.info(
            f"Turbulence Intensity has been calculated for the following measurements {columns}"
        )
        self.timeseries = df

    def calculate_upflow(self):
        """
        Caluclate the upflow angle of wind speed using top a/m and vertical anemometer
        The value is returned in rad from numpy but converted to degrees in the function
        """
        df = self.timeseries

        df["upflow"] = np.arctan(df["v_vert_Avg"] / df["v1_Avg"])
        df["upflow"] = df["upflow"] * 180 / np.pi

        log.info(f"The uplow angle has been calculated for {self.name}")

        self.timeseries = df

    def filter_timeseries_IEC(self, Filter):
        """
        Filtering met mast dataset based on IEC 61400-12-1 spesifications for site calibration
        1 (NumSamplesInterval_avg_PMM2 == 600)
        2 (DirNumSamples > 595)
        3 (Temperature > 2 C & Humidity < 80%)
        4 (Mean wind speed <= 4 & >= 16 m/s)
        5 (Measurement sector <=206.6 >=6.9
        """
        NumSamplesInterval = Filter["NumSamplesInterval"]
        WindAvgMin = Filter["WindAvgMin"][1]
        WindAvgMax = Filter["WindAvgMax"][1]
        DirAvgMin = Filter["DirAvgMin"][1]
        DirAvgMax = Filter["DirAvgMax"][1]
        DirNumSamples = Filter["DirNumSamples"]
        TempMin = Filter["TempMin"]
        HumidityMax = Filter["HumidityMax"]

        df = self.timeseries

        wind_col = Filter["WindAvgMin"][0]
        dir_col = Filter["DirAvgMin"][0]
        dir_columns = [i for i in df.columns if re.match("dir._num", i)]
        # wind_columns = [i for i in df.columns if re.match("v._Avg", i)]

        df["filter_samples"] = 0
        df["filter_samples_dir"] = 0
        df["filter_temp_hum"] = 0
        df["filter_wind"] = 0
        df["filter_sector"] = 0

        # NOTE this is number 1 filtering
        df.loc[df["NumSamplesInterval"] < NumSamplesInterval, "filter_samples"] = 1

        # NOTE this is number 2 filtering
        for dir_column in dir_columns:
            df.loc[df[dir_column] < DirNumSamples, "filter_samples_dir"] = 1

        # NOTE this is number 3 filtering
        df.loc[
            (df["thb_t_Avg"] < TempMin) | (df["thb_h_Avg"] > HumidityMax),
            "filter_temp_hum",
        ] = 1

        # NOTE this is number 4 filtering
        df.loc[
            ((df[wind_col] < WindAvgMin) | (df[wind_col] > WindAvgMax)), "filter_wind"
        ] = 1

        # NOTE this is number 5 filtering
        df.loc[
            ((df[dir_col] < DirAvgMin) | (df[dir_col] > DirAvgMax)), "filter_sector"
        ] = 1

        log.info(
            f"Filtering Base IEC has been applied {"\n".join(f"{k} -- {v}" for k, v in Filter.items())}"
        )

        self.timeseries = df

    def filter_timeseries_add(self, Filter):
        """
        Apply additional filter as this is detailed in the Stranoch reports
        1 (inflow_angle_PMM >= -2 & inflow_angle_PMM <= 2)
        2 (TI_V1_PMM >= 0.06 & TI_V1_PMM <= 0.24)
        3 (alpha_V1_V3_PMM >= 0 & alpha_V1_V3_PMM <= 0.5)
        4 (Precipitation_avg_PMM2 < 10)
        """
        InflowAngleMin = Filter["InflowAngleMin"]
        InflowAngleMax = Filter["InflowAngleMax"]
        TurbulanceIntencityMin = Filter["TurbulanceIntencityMin"]
        TurbulanceIntencityMax = Filter["TurbulanceIntencityMax"]
        AlphaMin = Filter["AlphaMin"]
        AlphaMax = Filter["AlphaMax"]
        PrecipitationMax = Filter["PrecipitationMax"]

        df = self.timeseries

        df["filter_inflow"] = 0
        df["filter_TI"] = 0
        df["filter_alpha"] = 0

        # NOTE this is number 1 filtering
        df.loc[
            (df["upflow"] < InflowAngleMin) | (df["upflow"] > InflowAngleMax),
            "filter_inflow",
        ] = 1

        # NOTE this is number 2 filtering
        df.loc[
            (df[TurbulanceIntencityMin[0]] < TurbulanceIntencityMin[1])
            | (df[TurbulanceIntencityMax[0]] < TurbulanceIntencityMax[1]),
            "filter_TI",
        ] = 1

        # NOTE this is number 3 filtering
        df.loc[
            (df["alpha"] < AlphaMin) | (df["alpha"] > AlphaMax),
            "filter_alpha",
        ] = 1

        # NOTE this is number 4 filtering
        df.loc[
            df["Precipitation"] > PrecipitationMax,
            "filter_inflow",
        ] = 1

        self.timeseries = df


class Site(BaseModel):
    id: int = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    name: str
    CMM: MetMast
    PMM: MetMast

    def site_calibration(self):
        """
        This is the main method that will perform site calibrations
        """
        pass
