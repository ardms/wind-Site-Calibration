import re
import pandas as pd
import numpy as np
import logging
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
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
    """
    A class representing a meteorological mast (Met Mast) with its associated data and methods for processing timeseries data.

    Attributes:
        id (int): Unique ID based on current timestamp.
        name (str): Name of the Met Mast.
        date_commission (datetime, optional): Commission date of the Met Mast.
        anemometers (dict): Details of anemometers.
        vanes (dict): Details of wind vanes.
        thermometers (dict): Details of thermometers.
        presipitation_names (list, optional): List of precipitation sensor names.
        presipitation_heights (list, optional): List of precipitation sensor heights.
        data_type (str, optional): Type of data files (e.g., 'csv').
        timeseries_path (Path): Path to the directory containing timeseries data files.
        timeseries_skiprows (list, optional): Rows to skip when reading timeseries data.
        timeseries_header (int, optional): Header row index for timeseries data.
        timeseries_index_col (str, optional): Column to set as index in timeseries data.
        timeseries (list, optional): List to store the timeseries data after processing.
    """

    id: int = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    name: str
    date_commission: datetime | None = None
    anemometers: dict
    vanes: dict
    thermometers: dict
    presipitation_names: list | None = None
    presipitation_heights: list | None = None
    data_type: str | None = None
    timeseries_path: Path
    timeseries_skiprows: list | None = None
    timeseries_header: int | None = None
    timeseries_index_col: str | None = None
    timeseries: list | None = None

    def load_timeseries_file(self, file):
        """
        Read a timeseries CSV file into a DataFrame with specified parameters.

        Args:
            file (str or Path): Path to the CSV file to be read.

        Returns:
            DataFrame: The timeseries data converted to float64 data type.
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
        Read all timeseries data files from a specified folder, merge them into a single DataFrame,
        and apply calibration corrections (slopes and offsets) to the instrument data.
        """
        folder = Path(self.timeseries_path)
        list_of_dfs = []
        for file in folder.glob(f"*.{self.data_type}"):
            log.info(f"Working on {file.name}")
            df = self.load_timeseries_file(file)
            list_of_dfs.append(df)

        df_concat = pd.concat(list_of_dfs)

        # NOTE this little snippet will loop through all the relevant columns and apply slope and offset
        for instrument in [self.anemometers, self.vanes, self.thermometers]:
            for key in instrument:
                stem = instrument[key]["name"]
                columns = [stem + "_" + i for i in instrument[key]["suffix"]]
                for col in columns:
                    # print(col)
                    slope = instrument[key]["slope"]
                    offset = instrument[key]["offset"]
                    df_concat[col] = df_concat[col] * slope + offset
                    if instrument == self.vanes:
                        df_concat[col] = df_concat[col] % 360
        try:
            self.timeseries = df_concat
            log.info(f"All data from {folder} have been succesfully loaded")
        except Exception as ex:
            log.error(f"{ex} timeseries has not been updated")

    def calculate_alpha(self, low_wind: (str, int), high_wind: (str, int)):
        """
        Calculate the wind shear exponent (alpha) based on wind speed measurements at two different heights.
        Formula Vzi = Vh*(Zi/H)^a

        Args:
            low_wind (tuple): A tuple containing the name of the column for lower height wind speed and its height.
            high_wind (tuple): A tuple containing the name of the column for higher height wind speed and its height.

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
        Calculate Turbulence Intensity (TI) for each anemometer as the ratio of the standard deviation to the mean wind speed.
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

    def calculate_dir_bins(self, Filter):
        """
        Calculate directional bins based on wind direction data. As per IEC 61400-12-1 width of the bin is 10 Deg.

        Args:
            Filter (dict): A dictionary containing filtering parameters.
        """
        df = self.timeseries
        df["dir_bin"] = np.nan
        Dir = Filter["Dir"]
        for key, section in Dir.items():
            DirAvgMin = section["Min"]
            DirAvgMax = section["Max"]
            dir_col = section["column_name"]
            DirMin = 5 * round(DirAvgMin / 5)
            if DirMin > DirAvgMax:
                DirMax = 360 + DirAvgMax
                rr = [i % 360 for i in range(int(DirMin), int(DirMax), 10)]
                ii = 0
                while ii < len(rr) - 1:
                    angle_min = rr[ii]
                    ii += 1
                    angle_max = rr[ii]
                    df.loc[
                        (df[dir_col] > angle_min) & (df[dir_col] <= angle_max),
                        "dir_bin",
                    ] = angle_min
            else:
                DirMax = DirAvgMax
                rr = [i % 360 for i in range(int(DirMin), int(DirMax), 10)]
                ii = 0
                while ii < len(rr) - 2:
                    angle_min = rr[ii]
                    ii += 1
                    angle_max = rr[ii]
                    df.loc[
                        (df[dir_col] > angle_min) & (df[dir_col] <= angle_max),
                        "dir_bin",
                    ] = angle_min
                df.loc[(df[dir_col] > rr[-2]) | (df[dir_col] <= rr[-1]), "dir_bin"] = (
                    rr[-2]
                )
        self.timeseries = df

    def filter_timeseries_IEC(self, Filter):
        """
        Filtering met mast dataset based on IEC 61400-12-1 spesifications for site calibration
        1 (NumSamplesInterval_avg_PMM2 == 600)
        2 (DirNumSamples > 595)
        3 (Temperature > 2 C & Humidity < 80%)
        4 (Mean wind speed <= 4 & >= 16 m/s)
        5 (Measurement sector <=DirMin >=DirMax

        Args:
            Filter (dict): A dictionary containing filtering parameters.
        """
        NumSamplesInterval = Filter["NumSamplesInterval"]
        WindAvgMin = Filter["WindAvgMin"][1]
        WindAvgMax = Filter["WindAvgMax"][1]
        DirNumSamples = Filter["DirNumSamples"]
        TempMin = Filter["TempMin"]
        HumidityMax = Filter["HumidityMax"]
        Dir = Filter["Dir"]

        df = self.timeseries

        wind_col = Filter["WindAvgMin"][0]
        temp_col = Filter["TempMin"][0]
        try:
            humidity_col = Filter["HumidityMax"][0]
        except Exception as ex:
            log.error(f"{ex}")
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
        if HumidityMax:
            df.loc[
                (df[temp_col] < TempMin[1]) | (df[humidity_col] > HumidityMax[1]),
                "filter_temp_hum",
            ] = 1
        else:
            df.loc[
                (df[temp_col] < TempMin[1]),
                "filter_temp_hum",
            ] = 1

        # NOTE this is number 4 filtering
        df.loc[
            ((df[wind_col] < WindAvgMin) | (df[wind_col] > WindAvgMax)), "filter_wind"
        ] = 1

        # NOTE this is number 5 filtering
        for key, aa in Dir.items():
            DirAvgMin = aa["Min"]
            DirAvgMax = aa["Max"]
            dir_col = aa["column_name"]
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

        Args:
            Filter (dict): A dictionary containing additional filtering parameters.
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
        if PrecipitationMax:
            df.loc[
                df["Precipitation"] > PrecipitationMax,
                "filter_inflow",
            ] = 1
        else:
            log.info("No Precipitation filtering has been applied")

        self.timeseries = df


class Site(BaseModel):
    """
    A class representing a site with multiple Met Masts and methods for combining and analyzing their data.

    Attributes:
        id (int): Unique ID based on current timestamp.
        name (str): Name of the Site.
        CMM (MetMast): Central Met Mast object.
        PMM (MetMast): Peripheral Met Mast object.
        joined_timeseries (list, optional): List to store the joined timeseries data after processing.
    """

    id: int = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    name: str
    CMM: MetMast
    PMM: MetMast
    joined_timeseries: list | None = None

    def join_timeseries(self):
        """
        Join the data from both met masts into one combined dataframe.
        """
        df = self.PMM.timeseries.join(
            self.CMM.timeseries, lsuffix="_PMM", rsuffix="_CMM", how="outer"
        )
        self.joined_timeseries = df

    def linear_regression(self):
        """
        Linear regression using
        """
        pass

    def site_calibration(self):
        """
        This is the main method that will perform site calibrations
        """
        pass
