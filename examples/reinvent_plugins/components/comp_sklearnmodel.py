"""
Test
"""

__all__ = ["SKLearnModel"]
from dataclasses import dataclass
from typing import List
import logging

import numpy as np
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    # descriptor: List[str]
    model_file: str
    is_regressor: bool
    ecfp_radius: int
    ecfp_size: int


@add_tag("__component")
class SKLearnModel:
    def __init__(self, params: Parameters):
        with open(params.model_file[0], "rb") as f:
            self.clf = pkl.load(f)

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=params.ecfp_radius[0],fpSize=params.ecfp_size[0])
        self.is_regressor = params.is_regressor[0]



    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:

        X = [self.mfpgen.GetFingerprint(s) for s in mols]
        if self.is_regressor:
            scores = self.clf.predict(X)
        else:
            scores = self.clf.predict_proba(X)

        return ComponentResults([scores])