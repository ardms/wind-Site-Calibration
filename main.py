# from textual.app import App
import pandas as pd
from pathlib import Path
import json
from modules.schema import MetMast, Site


with open("./config.json") as conf_file:
    conf = json.load(conf_file)

# data_path = Path(conf["properties"]["data"]["path"])

PMM_T03 = MetMast(name="PMM_T03", **conf["PMM"]["data"])

breakpoint()
