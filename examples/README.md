# Examples

Notebooks, configs, and REINVENT plugin to replicate the experiments in the manuscript.

## Prerequisites

- A working REINVENT4 installation. See the [REINVENT repository](https://github.com/MolecularAI/REINVENT4) for setup instructions, including how to register external plugin directories.
- The MOO scalarization methods used in this paper (`chebyshev_scalarization`, `generational_distance`, `non_dominated_sorting`, and `hit_rate`) are contributed to the official REINVENT repository. Consult the REINVENT repository for the latest availability.

## Notebooks

Run in order:

1. **`01_define_scaffolds.ipynb`** — Extracts scaffolds from the public PDE10A dataset and saves them as `.smi` files referenced by the REINVENT configs.

   Requires the external [`AutomatedSeriesClassification`](https://github.com/rdkit/AutomatedSeriesClassification) package and a pre-built ChEMBL substructure library pickle (`chembl33_sssdata.pkl`). Follow the setup instructions in the `AutomatedSeriesClassification` README to generate this file before running the notebook. Raw potency data must be obtained from the public sources cited in the manuscript.

2. **`02_train_objective_model.ipynb`** — Trains a random forest model for a chosen target or ADMET property and serialises it as a `.pkl` file consumed by the `SKLearnModel` scoring component. Set the `project` variable at the top of the notebook to select the target. Models for each scoring component used in the paper must be trained separately. QED and SA score are computed directly by REINVENT and do not require trained models.

   Raw property datasets must be obtained from the public sources cited in the manuscript.

## LibINVENT configs (`configs/`)

Five TOML configs for LibINVENT corresponding to 5 independent replicates of the Chebyshev scalarization MOO method on scaffold 13. Each file differs only in the random `seed`. These serve as templates — adapt the following fields to reproduce the full set of experiments described in the paper:

| Field | What to change |
|---|---|
| `smiles_file` | Path to the scaffold `.smi` file for the target scaffold |
| `[stage.scoring] type` | MOO method (see table below) |
| `tb_logdir`, `json_out_config`, `summary_csv_prefix`, `chkpt_file` | Unique output names per replicate to avoid overwriting results |

### Supported MOO scoring types

| `type` value | Description |
|---|---|
| `arithmetic_mean` | Weighted arithmetic mean of component scores |
| `geometric_mean` | Weighted geometric mean of component scores |
| `chebyshev_scalarization` | Chebyshev (min-max) scalarization |
| `generational_distance` | Generational distance from a reference point |
| `non_dominated_sorting` | Non-dominated sorting (NSGA-II-style) |
| `hit_rate` | Fraction of components exceeding their threshold; use `weight` per component to set the threshold |

All configs assume they are run from the directory containing the scaffold `.smi` files and trained `.pkl` models, and that a GPU is available at `cuda:0`. Change `device` to `cpu` or another CUDA index if needed.

## REINVENT plugin (`reinvent_plugins/`)

`components/comp_sklearnmodel.py` is a custom scoring component that wraps a serialised scikit-learn model. It accepts the following parameters in the TOML config:

| Parameter | Description |
|---|---|
| `model_file` | Path to the `.pkl` model |
| `is_regressor` | `true` for regressors, `false` for classifiers |
| `ecfp_radius` | Morgan fingerprint radius |
| `ecfp_size` | Fingerprint bit size |

Register the `reinvent_plugins/` directory following the plugin path instructions in the REINVENT README before running.
