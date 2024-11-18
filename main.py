# -*- coding: utf-8 -*-
"""
Created on 2024/11/18

@author: adimopoulos
"""

# from textual.app import App
from pathlib import Path
import json
import logging
from modules.schema import MetMast, Site
from rich.logging import RichHandler
from rich.prompt import Prompt


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")


with open("./config.json") as conf_file:
    conf = json.load(conf_file)

PMM_T03 = MetMast(name="PMM_T03", **conf["PMM"]["data"])
PMM_T03.load_timeseries_from_folder()
