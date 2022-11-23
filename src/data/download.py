import os

from ..utils.general import get_base_dir

#get the base directory of the framework
RACE_C_DIR = os.path.join(get_base_dir(), 'data', 'RACE-C')
RACE_C_REPO = 'https://github.com/mrcdata/race-c/raw/master/data.zip'

#== Automatic downloading utils ===================================================================#
def download_race_plus_plus():
    """ automatically downloads the classification CAD datasets in parent data folder"""
    os.system(f"mkdir -p {RACE_C_DIR}")
    os.system(f"wget -O RACE-C.zip {RACE_C_REPO}")
    os.system(f"unzip RACE-C.zip -d {RACE_C_DIR}")
    os.system(f"mv {RACE_C_DIR}/data/* {RACE_C_DIR}")
    os.system(f"rmdir {RACE_C_DIR}/data")
    os.system(f"rm RACE-C.zip")
