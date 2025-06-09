import os
import pandas as pd
from pycrystallography.tran_texture.generate_simulated_odf import generate_simulated_odf


def test_generate_odf(tmpdir):
    params = {
        "crystal_symmetry": "hexagonal",
        "odf_grid_spacing": 10,
        "angle_units": "degree",
        "output_file": os.path.join(tmpdir, "odf.csv"),
    }
    path = generate_simulated_odf(params)
    assert os.path.isfile(path)
    df = pd.read_csv(path)
    assert df.shape[1] == 4
