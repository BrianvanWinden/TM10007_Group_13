import pandas as pd
import os
from zipfile import ZipFile


def load_data():
    this_directory = os.path.dirname(os.path.abspath(_file_))
    zip_dir = os.path.join(this_directory, 'ecg_data.zip')
    zf = ZipFile(zip_dir, "r")
    zf.extractall(this_directory)
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    return  data









