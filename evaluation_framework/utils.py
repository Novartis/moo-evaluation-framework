"""
Utility functions for molecular filtering and scaffold matching.

This module provides helper functions for:
- Checking NIBR filter strictness
- Matching scaffolds against known molecule databases
"""

from typing import Generator, List, Literal, Tuple
import logging
import os

import numpy as np
from rdkit import Chem
import pandas as pd

log = logging.getLogger(__name__)


def check_strict_filters(filter: List) -> bool:
    """
    Check if a molecule passes strict NIBR filter criteria.

    Evaluates NIBR filter results to determine if a molecule meets strict
    medicinal chemistry criteria. Fails molecules with:
    - Invalid filter data (float/NaN)
    - Substructure matches (not 'no match')
    - Covalent flags
    - Special molecule flags
    - High severity scores (≥10)

    Args:
        filter (List): NIBR filter result list with structure:
            [0]: NaN or warning
            [1]: Number of heavy atoms
            [2]: Match status ('no match' if clean)
            [3]: Fraction N/O
            [4]: Covalent flag (0 or 1)
            [5]: Special molecule flag (0 or 1)
            [6]: Severity score (0-10)

    Returns:
        bool: True if molecule passes all strict criteria, False otherwise

    Example:
        >>> filter_result = [np.nan, 15, 'no match', 0.2, 0, 0, 2]
        >>> check_strict_filters(filter_result)
        True
        >>> filter_result = [np.nan, 15, 'PAINS', 0.2, 0, 0, 8]
        >>> check_strict_filters(filter_result)
        False
    """
    if isinstance(filter, float):
        return False
    if filter[2] != "no match":
        return False
    if filter[4] == 1:
        return False
    if filter[5] == 1:
        return False
    if filter[-1] == 10:
        return False
    return True


def get_scaffold_matches(
    scaffold_path: str, knownmolecules_path: str
) -> Generator[Tuple[int, List[bool]], None, None]:
    """
    Match scaffolds against known molecule database.

    Reads scaffold files (.smi format) and checks which known molecules
    contain each scaffold as a substructure. Removes dummy atoms (*) 
    from scaffolds before matching.

    Args:
        scaffold_path (str): Directory path containing scaffold .smi files.
            Files should be named with pattern ending in '_N.smi' where N
            is the scaffold identifier number.
        knownmolecules_path (str): Path to CSV file with 'SMILES' column
            containing known molecule SMILES strings.

    Yields:
        Tuple[int, List[bool]]: 
            - Scaffold identifier (int) extracted from filename
            - Boolean list indicating which known molecules contain the scaffold

    Example:
        >>> for scf_id, matches in get_scaffold_matches('./scaffolds', './known.csv'):
        ...     print(f"Scaffold {scf_id}: {sum(matches)} matches")
        Scaffold 1: 45 matches
        Scaffold 2: 23 matches

    Note:
        - Only processes files ending with .smi extension
        - Removes atoms with atomic number 0 (dummy atoms/placeholders)
        - Requires known molecules CSV to have 'SMILES' column
    """
    for scf_f in os.listdir(scaffold_path):
        if not scf_f.endswith(".smi"):
            continue
            
        with open(os.path.join(scaffold_path, scf_f)) as f:
            scf = f.read()
        scf = Chem.MolFromSmiles(scf)

        # Remove dummy atoms (marked with *)
        dummy_info = [
            (atom.GetIdx(), atom.GetAtomMapNum())
            for atom in scf.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]

        editable_mol = Chem.EditableMol(scf)
        for idx, _ in sorted(dummy_info, reverse=True):
            editable_mol.RemoveAtom(idx)
        scf = editable_mol.GetMol()

        # Load known molecules and check for substructure matches
        km = pd.read_csv(knownmolecules_path)
        nm = scf_f.split("_")[-1]
        nm = int(nm.split(".smi")[0])
        
        matches = [
            Chem.MolFromSmiles(x).HasSubstructMatch(scf) if Chem.MolFromSmiles(x) else False
            for x in km["SMILES"]
        ]
        
        yield nm, matches


