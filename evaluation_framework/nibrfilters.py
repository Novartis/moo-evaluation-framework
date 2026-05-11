"""
From original nibr filters implementation in RDKit

"""
import argparse
import operator
import sys
from collections import Counter, defaultdict, namedtuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import FilterCatalog, RDConfig, rdMolDescriptors

FilterMatch = namedtuple(
  'FilterMatch',
  ('SubstructureMatches', 'NumHeavyAtoms', 'Min_N_O_filter', 'Frac_N_O', 'Covalent', 'SpecialMol', 'SeverityScore'))


# Build the filter catalog using the RDKit filterCatalog module
def buildFilterCatalog():

  inhousefilter = pd.read_csv(
    f'{RDConfig.RDContribDir}/NIBRSubstructureFilters/SubstructureFilter_HitTriaging_wPubChemExamples.csv'
  )
  inhouseFiltersCat = FilterCatalog.FilterCatalog()

  inhousefilter = inhousefilter.reset_index(drop = True)
  for i in range(inhousefilter.shape[0]):
    mincount = 1
    if inhousefilter['MIN_COUNT'][i] != 0:
      mincount = int(inhousefilter['MIN_COUNT'][i])
    pname = inhousefilter['PATTERN_NAME'][i]
    sname = inhousefilter['SET_NAME'][i]
    pname_final = '{0}_min({1})__{2}__{3}__{4}'.format(pname, mincount,
                                                       inhousefilter['SEVERITY_SCORE'][i],
                                                       inhousefilter['COVALENT'][i],
                                                       inhousefilter['SPECIAL_MOL'][i])
    fil = FilterCatalog.SmartsMatcher(pname_final, inhousefilter['SMARTS'][i], mincount)
    inhouseFiltersCat.AddEntry(FilterCatalog.FilterCatalogEntry(pname_final, fil))
    inhouseFiltersCat.GetEntry(i).SetProp('Scope', sname)
  return inhouseFiltersCat

def assignFilters(catalog, smi):

  inhouseFiltersCat = catalog

  NO_filter = '[#7,#8]'
  sma = Chem.MolFromSmarts(NO_filter, mergeHs=True)

  
    
  qc, NO_filter, fracNO, co, sc, sm = [np.nan] * 6

# The following files require numpy and were explicitely checked for their compatibility.
  try:
    mol = Chem.MolFromSmiles(smi)

    # fraction of N and O atoms
    numHeavyAtoms = mol.GetNumHeavyAtoms()
    numNO = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7,#8]')))
    fracNO = float(numNO) / numHeavyAtoms

    # special NO filter
    if not mol.HasSubstructMatch(sma):
      NO_filter = 'no_oxygen_or_nitrogen'
    else:
      NO_filter = 'no match'

    # all substructure filters
    entries = inhouseFiltersCat.GetMatches(mol)
    if len(list(entries)):
      # initialize empty lists
      fs, sev, cov, spm = ([] for _ in range(4))
      # get the matches
      for entry in entries:
        pname = entry.GetDescription()
        n, s, c, m = pname.split('__')
        fs.append(entry.GetProp("Scope") + '_' + n)
        sev.append(int(s))
        cov.append(int(c))
        spm.append(int(m))
      # concatenate all matching filters
      qc = ' | '.join(fs)
      # assign overall severity
      if sev.count(2):
        sc = 10
      else:
        sc = sum(sev)
      # get number of covalent flags and special molecule flags
      co = sum(cov)
      sm = sum(spm)
    # if non of the filters matches
    else:
      qc = 'no match'
      sc = 0
      co = 0
      sm = 0

    
  except Exception:
    print("Failed on compound {0}\n".format(smi))
    pass
  return FilterMatch(qc, numHeavyAtoms, NO_filter, fracNO, co, sm, sc)


import os
from tqdm import tqdm


def apply_nibrfilters(folder_path):
    
    if os.path.exists(os.path.join(folder_path, "scalarization_methods_all.csv")):
            print("Using precalculated dataframe")
            scalarization_methods = pd.read_csv(os.path.join(folder_path, "scalarization_methods_all.csv"))
    else:
       print("DataFrame not found")
       exit()
    
    smiles = scalarization_methods[(~scalarization_methods["SMILES"].str.startswith("INVALID"))]["SMILES"].unique()
    for x in range(1, 50000):
        if len(smiles) % x ==0:
            chunk_size = x

    print(f"Using chunk size: {chunk_size}")

    catalog = buildFilterCatalog()

    nibr_scores = []
    for chunk in tqdm(np.split(list(smiles), np.arange(chunk_size, len(smiles), chunk_size))):
        nibr_scores.extend(list(map(lambda x: assignFilters(catalog, x), chunk)))
    nibr_scores_smiles = {smi: list(scr) for smi, scr in zip(smiles, nibr_scores)}
    scalarization_methods["NIBRFilters"] = scalarization_methods["SMILES"].map(nibr_scores_smiles)

    scalarization_methods.to_csv(os.path.join(folder_path, "scalarization_methods_all.csv"), index= None)




if __name__ == "__main__":
    '''
    example --> python run_nibrfilters.py --folder_path /path/to/folder
    '''

    parser = argparse.ArgumentParser(
        description="Calculate NIBR filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to folder with scalarization framework",
    )
    
    args = parser.parse_args()

    
    apply_nibrfilters(args.folder_path)

