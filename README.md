# Multi-Objective Optimization Evaluation Framework

A Python framework for evaluating and comparing multi-objective optimization (MOO) algorithms in molecular design and drug discovery, developed in support of the accompanying manuscript.

## Overview

This framework provides a systematic approach to analyze MOO algorithm performance across multiple dimensions:

- **Chemical validity**: SMILES validation and structural filters
- **Chemical diversity**: Tanimoto distance metrics and uniqueness analysis
- **Multi-objective performance**: Pareto optimality, hypervolume indicators
- **Novelty assessment**: Comparison against known molecule databases
- **Drug-likeness**: NIBR medicinal chemistry filter compliance
- **Target achievement**: Hit rate analysis against Target Property Profiles (TPP)

## Key Features

- **Flexible configuration**: Modular configuration system for data, analysis, and visualization
- **Comprehensive metrics**: 15+ performance indicators including hypervolume, diversity, and novelty
- **Publication-quality visualizations**: Box plots, radar charts, Pareto front analysis, and comparative scatter plots
- **Statistical analysis**: Automatic statistical testing with significance annotation
- **Efficient computation**: Caching and chunked processing for large datasets
- **Extensible design**: Easy addition of custom metrics and objectives

## Installation

### Prerequisites

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) for environment and dependency management

Install uv if you haven't already. Find the installation process [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/Novartis/moo-evaluation-framework.git
cd moo_evaluation_framework
```

2. **Create and activate the virtual environment:**

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install the package and dependencies (locked versions for reproducibility):**

```bash
uv sync --frozen
```

4. **Verify the installation:**

```bash
python -c "import evaluation_framework"
```

### Key Dependencies

All dependencies are automatically installed via uv:

| Package | Version |
|---------|---------|
| pandas | ≥ 2.3.2 |
| numpy | < 2 |
| matplotlib | ≥ 3.10.5 |
| seaborn | ≥ 0.13.2 |
| scipy | ≥ 1.16.1 |
| rdkit | == 2022.9.5 |
| pygmo | ≥ 2.19.5 |
| scikit-learn | ≥ 1.7.2 |
| tqdm | ≥ 4.67.1 |

## Quick Start

```python
from evaluation_framework.main import MOOEvaluation
from evaluation_framework.config import AnalysisConfig, DataConfig, PlotConfig

# 1. Configure analysis parameters
analysis_config = AnalysisConfig(
    objectives=['QED', 'SA_Score', 'LogP'],
    comparison_columns=['Scaffold', 'MOO', 'Replicate']
)

# 2. Configure data source
data_config = DataConfig(
    data_path='path/to/moo_results.csv',
    smiles_col='SMILES',
    objectives=['QED', 'SA_Score', 'LogP'],
    comparison_columns=['Scaffold', 'MOO', 'Replicate'],
    moo_col='MOO',
    epoch_col='Step',
    replicate_col='Replicate'
)

# 3. Configure visualization
plot_config = PlotConfig(
    objectives=['QED', 'SA_Score', 'LogP'],
    plots_path='./results/plots',
    dpi=300,
    moo_order=['ArithmeticMean', 'GeometricMean', 'ChebyshevScalarization']
)

# 4. Initialize evaluator
evaluator = MOOEvaluation(analysis_config)
evaluator.prep_data(data_config)
evaluator.prep_plots(plot_config)

# 5. Apply Target Product Profile thresholds
evaluator.apply_tpp(0.5)

# 6. Generate comprehensive evaluation
summary = evaluator.generate_evaluation_summary()

# 7. Create visualizations
evaluator.evaluation_visual(
    eval_columns=['Hypervolume Pareto Molecules', 'Minimum Novelty', 'Match TPP 0.5'],
    plot_type='box_reduced',
    share_axis=True
)
```

## Module Structure

```
moo_evaluation_framework/
├── README.md                  # This file
├── examples/                  # Notebooks and configs to reproduce paper experiments
├── evaluation_framework/
│   ├── __init__.py
│   ├── main.py                # High-level orchestration (MOOEvaluation)
│   ├── analysis.py            # Core metric calculations (MOOData)
│   ├── visualise.py           # Plotting and visualization (MOOPlots)
│   ├── config.py              # Configuration dataclasses
│   ├── nibrfilters.py         # NIBR medicinal chemistry filters
│   └── utils.py               # Utility functions
├── pyproject.toml
└── uv.lock
```

### Core Classes

#### `MOOEvaluation` (main.py)
High-level orchestrator coordinating the complete analysis pipeline. Manages data loading, metric calculation, and visualization generation.

**Key Methods:**
- `prep_data(config)`: Load and validate MOO results
- `apply_tpp(thresholds)`: Set Target Product Profile thresholds
- `generate_evaluation_summary(metrics)`: Calculate all performance metrics
- `evaluation_visual(eval_columns, plot_type)`: Generate comparative visualizations
- `prep_plots(config)`: Establish settings for visualization generator

#### `MOOData` (analysis.py)
Data management and metric calculation engine. Handles molecule validation, filter application, and computation of diversity, Pareto, and novelty metrics.

**Key Methods:**
- `get_pareto_molecules()`: Identify Pareto-optimal solutions
- `get_hypervolume()`: Calculate hypervolume indicator
- `get_internal_tanimoto_distance()`: Measure chemical diversity
- `get_novelty_tanimoto_distance()`: Assess novelty vs. known molecules
- `get_nondominated_solutions()`: Compare against reference compounds

#### `MOOPlots` (visualise.py)
Visualization generator producing publication-quality plots. Supports multiple plot types and automatic statistical testing.

**Key Methods:**
- `plot_evaluation_summary()`: Multi-metric comparison plots
- `plot_molecule_summary()`: Quality filtering cascade
- `create_comparison_plot()`: Head-to-head method comparisons

## Input Data Format

### Required CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| SMILES | str | Molecular structure | `'CCO'` |
| Objectives | float | Optimization targets (3+) | `0.85` |
| MOO | str | Algorithm identifier | `'ArithmeticMean'` |
| Scaffold | str/int | Molecular scaffold ID | `'benzene'` |
| Replicate | int | Replicate number | `1` |
| Step/Epoch | int | Optimization iteration | `100` |
| NIBRFilters | str | Pre-calculated filter results (optional) | `"[...]"` |

## Available Metrics

### Quality Metrics
- **Valid Molecules**: Proportion with valid SMILES
- **Unique Molecules**: Proportion after deduplication
- **Pass NIBR Filters**: Drug-likeness compliance rate

### Property Metrics
- **Match TPP X.X**: Hit rate at threshold (0.3, 0.5, 0.7)
- **Proportion Pareto Molecules**: Fraction on Pareto front
- **Hypervolume Pareto Molecules**: Dominated objective space volume
- **Similarity Ideal Vector**: Distance to ideal point
- **Nondominated Solutions**: Solutions not dominated by references

### Chemical Structure Metrics
- **Average TD per step**: Chemical breadth evolution
- **Internal Tanimoto Distance**: Within-method diversity
- **Unique MOO Molecules**: Method-exclusive discoveries
- **Minimum Novelty**: Distance to known molecules

## Visualization Types

### Box Plots (`plot_type='box_reduced'`)
Distribution comparison across methods with significance statistics.

```python
evaluator.evaluation_visual(
    eval_columns=['Hypervolume Pareto Molecules', 'Match TPP 0.5'],
    plot_type='box_reduced',
    share_axis=True,
    p_values=[0.05, 0.01, 0.001]
)
```

### Radar Charts (`plot_type='radar'`)
Multi-metric profile comparison (recommended for ≤8 metrics).

```python
evaluator.evaluation_visual(
    eval_columns=['Hypervolume', 'Diversity', 'Novelty', 'HitRate'],
    plot_type='radar'
)
```

### Head-to-Head Comparisons
Pairwise scatter plots showing win rates.

```python
from evaluation_framework.visualise import MOOPlots

plotter = MOOPlots(plot_config)
# generate_evaluation_summary() returns a multi-index DataFrame; reset before plotting
summary_flat = summary.reset_index()
for grid in plotter.create_comparison_plot(summary_flat, 'Hypervolume Pareto Molecules', 'Scaffold'):
    grid.savefig(f'comparison_{moo_method}.png')
```

## Advanced Usage

### Custom Molecule Sets

Define specific molecules for analysis:

```python
# Set known molecules for novelty calculation
known_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']
evaluator.set_known_molecules(known_smiles)

# Set reference properties for non-dominated analysis
known_properties = {
    ('scaffold_A', None): np.array([[0.8, 0.7, 0.6], [0.7, 0.8, 0.5]]),
    ('scaffold_B', None): np.array([[0.75, 0.65, 0.55]])
}
evaluator.set_known_molecule_properties(known_properties)

# Define custom relevant molecules
relevant = (df['QED'] > 0.5) & (df['SA_Score'] > 0.4)
evaluator.define_relevant_molecules(relevant)
```

### Selective Metric Calculation

Optimize performance for large datasets:

```python
# Calculate only specific metrics
summary = evaluator.generate_evaluation_summary(
    metrics=['Hypervolume Pareto Molecules', 'Minimum Novelty', 'Match TPP 0.5']
)
```

### Custom Thresholds

```python
# Single threshold for all objectives
evaluator.apply_tpp(0.5)

# Per-objective thresholds (list)
evaluator.apply_tpp([0.5, 0.6, 0.4])

# Named objective thresholds (dict)
evaluator.apply_tpp({'QED': 0.5, 'SA_Score': 0.6, 'LogP': 0.4})
```

## Performance Considerations

- **Caching**: Fingerprint calculations are cached using `@lru_cache`
- **Chunking**: Large SMILES lists processed in optimized chunks
- **Lazy loading**: Metrics calculated only when requested
- **Column selection**: Load only required columns with `load_all_columns=False`

For datasets >100,000 molecules, pre-calculate NIBRFilters separately:

```bash
python -m evaluation_framework.nibrfilters --folder_path /path/to/data
```

## Statistical Testing

All comparative plots include automatic one-way statistical testing with customizable significance thresholds:

```python
evaluator.evaluation_visual(
    eval_columns=['Hypervolume'],
    plot_type='box_reduced',
    p_values=[0.05, 0.01, 0.001, 0.0001]  # Significance levels
)
```

Results annotated as:
- `p < 0.05`: Significant (*)
- `p < 0.01`: Highly significant (**)
- `p < 0.001`: Very highly significant (***)
- `NS`: Not significant (p ≥ 0.05)

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{TODO,
  title   = {},
  author  = {},
  journal = {},
  year    = {},
  doi     = {}
}
```

## Acknowledgments

This framework builds on:
- **RDKit**: Open-source cheminformatics toolkit
- **PyGMO**: Multi-objective optimization algorithms and utilities
- **NIBR Filters**: Substructure filters from RDKit

