from __future__ import annotations

"""Utilities for generating product texture from a parent ODF."""

from pathlib import Path
from typing import List, Tuple

import logging

import numpy as np
import pandas as pd

from pycrystallography.core.orientationRelation import OrientationRelation as OriReln
from pycrystallography.core.millerPlane import MillerPlane
from pycrystallography.core.millerDirection import MillerDirection
from pycrystallography.core.orientation import Orientation

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ZRBETA_TO_ALPHA = {
    "parent": [
        {
            "PhaseName": "Beta",
            "symbol": "$\\beta",
            "cifName": "Zr-Beta.cif",
            "OR_Plane": [1, 1, 0],
            "OR_Direction": [1, -1, 1],
        }
    ],
    "products": [
        {
            "PhaseName": "Alpha",
            "symbol": "$\\alpha$",
            "cifName": "Alpha-ZrP63mmc.cif",
            "OR_Plane": [0, 0, 0, 1],
            "OR_Direction": [2, -1, -1, 0],
        }
    ],
}


class TransformationTextureSimulator:
    """Simulate transformation texture using an orientation relation."""

    def __init__(self, or_definition: dict, structure_dir: str | Path | None = None) -> None:
        self.or_definition = or_definition
        self.structure_dir = Path(structure_dir or "data/structureData")
        self.orientation_relation = self._build_orientation_relation()
        self.variants = self.orientation_relation.getVariants()[0]
        logger.info("Simulator initialized with %d variants", len(self.variants))

    def _build_orientation_relation(self) -> OriReln:
        parent = self.or_definition["parent"][0]
        product = self.or_definition["products"][0]
        parent_cif = self.structure_dir / parent["cifName"]
        product_cif = self.structure_dir / product["cifName"]
        names = [parent["PhaseName"], product["PhaseName"]]
        symbols = [parent["symbol"], product["symbol"]]
        structures = [str(parent_cif), str(product_cif)]
        st_parent, lat_parent = OriReln.getStructureFromCif(str(parent_cif))
        st_product, lat_product = OriReln.getStructureFromCif(str(product_cif))
        planes = [
            MillerPlane(hkl=parent["OR_Plane"], lattice=lat_parent),
            MillerPlane(hkl=product["OR_Plane"], lattice=lat_product),
        ]
        directions = [
            MillerDirection(vector=parent["OR_Direction"], lattice=lat_parent),
            MillerDirection(vector=product["OR_Direction"], lattice=lat_product),
        ]
        return OriReln(
            names=names,
            symbols=symbols,
            structures=structures,
            planes=planes,
            directions=directions,
            initiateVariants=True,
        )

    def generate_variants(
        self, parent_ori: Orientation, weight: float, selection: float
    ) -> List[Tuple[Orientation, float]]:
        """Return product variant orientations and weights for a parent."""
        n_total = len(self.variants)
        n_select = max(1, min(n_total, round(selection * n_total)))
        logger.debug(
            "Generating %d of %d variants (selection=%s)", n_select, n_total, selection
        )
        weight_each = weight / n_select
        selected = self.variants[:n_select]
        return [(parent_ori * var, weight_each) for var in selected]

    def transform_odf(
        self, input_file: str | Path, output_file: str | Path, selection: float = 1.0
    ) -> str:
        logger.info("Reading parent ODF from %s", input_file)
        df = pd.read_csv(input_file)
        logger.info("Processing %d parent orientations", len(df))
        rows = []
        for _, row in df.iterrows():
            p_ori = Orientation(euler=np.radians([row["phi1"], row["phi"], row["phi2"]]))
            variants = self.generate_variants(p_ori, row["volume fraction"], selection)
            for ori, wt in variants:
                phi1, phi, phi2 = ori.getEulerAngles(units="deg")
                rows.append({
                    "phi1": phi1,
                    "phi": phi,
                    "phi2": phi2,
                    "volume fraction": wt,
                })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_file, index=False)
        logger.info("Transformed ODF written to %s", output_file)
        return str(output_file)


def transform_texture_from_odf(
    input_file: str | Path,
    output_file: str | Path = "transformed_odf.csv",
    selection: float = 1.0,
) -> str:
    """Convenience function for transforming an ODF CSV file."""
    sim = TransformationTextureSimulator(ZRBETA_TO_ALPHA)
    return sim.transform_odf(input_file, output_file, selection)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate transformation texture")
    parser.add_argument("input", help="input ODF CSV file")
    parser.add_argument("--output", default="transformed_odf.csv", help="output CSV file")
    parser.add_argument(
        "--selection",
        type=float,
        default=1.0,
        help="fraction of variants that form (0-1)",
    )
    args = parser.parse_args()
    path = transform_texture_from_odf(args.input, args.output, args.selection)
    logger.info("Transformed ODF written to %s", path)
