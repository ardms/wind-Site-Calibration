# -*- coding: utf-8 -*-
"""
Created on 2024/11/18

@author: adimopoulos
"""

# from textual.app import App
import logging
import json
from rich.logging import RichHandler
from modules.schema import MetMast, Site


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

with open("./config.json") as conf_file:
    conf = json.load(conf_file)

PMM_T03 = MetMast(
    name="PMM_T03",
    anemometers=conf["PMM"]["anemometers"],
    vanes=conf["PMM"]["vanes"],
    thermometers=conf["PMM"]["thermometers"],
    **conf["PMM"]["data"]
)
PMM_T03.load_timeseries_from_folder()

PMM_T03.calculate_alpha(
    conf["PMM"]["filter"]["AlphaLow"], conf["PMM"]["filter"]["AlphaHigh"]
)
PMM_T03.calculate_TI()
PMM_T03.calculate_upflow()

PMM_T03_filter = conf["PMM"]["filter"]
PMM_T03.filter_timeseries_IEC(PMM_T03_filter)
PMM_T03.filter_timeseries_add(PMM_T03_filter)
breakpoint()
