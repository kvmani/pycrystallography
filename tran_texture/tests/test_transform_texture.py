import os
import pandas as pd
from tran_texture.generate_simulated_odf import generate_simulated_odf
from tran_texture.transform_texture import transform_texture_from_odf


def test_transform_texture(tmpdir):
    params = {
        "crystal_symmetry": "hexagonal",
        "odf_grid_spacing": 10,
        "angle_units": "degree",
        "output_file": os.path.join(tmpdir, "odf.csv"),
    }
    input_path = generate_simulated_odf(params)
    out_file = os.path.join(tmpdir, "transformed.csv")
    result = transform_texture_from_odf(input_path, out_file, selection=0.5)
    assert os.path.isfile(result)
    df = pd.read_csv(result)
    assert df.shape[1] == 4
