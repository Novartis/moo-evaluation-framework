from rdkit import Chem
from rdkit.Chem import FilterCatalog, RDConfig, rdMolDescriptors
from collections import Counter, defaultdict, namedtuple
import pandas as pd


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
    return None
    pass
  return FilterMatch(qc, numHeavyAtoms, NO_filter, fracNO, co, sm, sc)

def check_strict_filters(filter):
    if type(filter) == float:
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


### From Prefer
def filter_and_normalize_smiles(smiles):

    uncharger = rdMolStandardize.Uncharger()
    # format is smarts, then flag to allow one to pick and choose which to include
    forbidden_elements = Chem.MolFromSmarts(
        "[Cu,Sb,As,Sn,Pt,Te,Pd,Lu,Ge,Zn,Cu,Co,Ni,Fe,Hg,Zr,Mn,Ag,Bi,Cd,Cr,Ti,Al,Au,Mo,V,Mg,In,Ga,Pb,Ca,W]"
    )
    remover = SaltRemover.SaltRemover()

    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception as e:
        return None

    if mol is None or mol.GetNumAtoms() > 100:
        return None

    res, deleted = remover.StripMolWithDeleted(mol)
    # add a flag in case you want to remove or keep the salts - Default we keep it without the salt
    if len(deleted) != 0:
        return None

    mol = uncharger.uncharge(mol)
    if mol is None:
        return None

    if Chem.SanitizeMol(mol, catchErrors=True):  # maybe the molecule has changed
        return None

    if mol.HasSubstructMatch(forbidden_elements):
        return None

    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    if len(smi) < 2 or "*" in smi or "R" in smi:
        return None

    return smi
