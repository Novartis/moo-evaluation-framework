"""
Configuration dataclasses for MOO evaluation framework.

This module defines configuration objects for data loading, analysis setup,
and visualization parameters. Using dataclasses ensures type safety and
provides clear documentation of required parameters.

Classes:
    AnalysisConfig: Analysis parameters and comparison settings
    DataConfig: Data source and column mappings
    PlotConfig: Visualization settings and output configuration
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnalysisConfig:
    """
    Configuration for MOO analysis parameters.

    Attributes:
        objectives (List[str]): Names of objective properties to optimize
            (e.g., ['QED', 'SA_Score', 'LogP'])
        comparison_columns (List[str]): Columns for grouping analyses
            (e.g., ['Scaffold', 'MOO', 'Replicate'])
    """

    objectives: List[str]
    comparison_columns: List[str]


@dataclass
class DataConfig:
    """
    Configuration for data loading and column mappings.

    Attributes:
        data_path (str): Path to CSV file containing MOO results
        smiles_col (str): Column name for SMILES molecular representations
        objectives (List[str]): Names of objective property columns
        comparison_columns (Optional[List[str]]): Grouping columns for analysis.
            If None, creates default grouping column.
        invalid_str (Optional[str]): Prefix string identifying invalid SMILES.
            If None, uses RDKit parsing only. Defaults to None.
        replicate_col (Optional[str]): Column name for replicate identifiers.
            Required for uniqueness calculations. Defaults to None.
        moo_col (Optional[str]): Column name for MOO method identifiers.
            Required for method comparisons. Defaults to None.
        epoch_col (Optional[str]): Column name for optimization iterations/steps.
            Required for temporal analysis. Defaults to None.
        score_col (Optional[str]): Column name for scalar scores.
            Optional, used for additional metrics. Defaults to None.
        load_all_columns (bool): Whether to load all CSV columns (True) or
            only required columns (False). Set True for exploratory analysis,
            False for performance. Defaults to False.
    """

    data_path: str
    smiles_col: str
    objectives: List[str]
    comparison_columns: Optional[List[str]]
    invalid_str: Optional[str] = None
    replicate_col: Optional[str] = None
    moo_col: Optional[str] = None
    epoch_col: Optional[str] = None
    score_col: Optional[str] = None
    load_all_columns: bool = False


@dataclass
class PlotConfig:
    """
    Configuration for visualization generation.

    Attributes:
        objectives (List[str]): Names of objective properties (for axis labels)
        plots_path (Optional[str]): Directory for saving plots. If None,
            figures are returned instead of saved. Defaults to None.
        dpi (int): Resolution in dots per inch for saved figures.
            Typical values: 72 (screen), 150 (draft), 300 (publication).
            Defaults to 200.
        moo_order (Optional[List[str]]): Preferred plotting order for MOO methods.
            If None, uses natural DataFrame ordering. Defaults to None.
        comparison_columns (Optional[List[str]]): Grouping columns for plots.
            Should match DataConfig.comparison_columns. Defaults to None.
        comparison_col (Optional[str]): Primary column for statistical tests.
            If None, uses first entry in comparison_columns. Defaults to None.
        smiles_col (str): Column name for SMILES strings. Should match
            DataConfig.smiles_col. Defaults to 'SMILES'.
        include_random (bool): Whether to include random baseline in plots.
            Adds reference lines for random performance. Defaults to False.
        random_label (str): Label identifying random baseline in data.
            Used to filter baseline from box plots. Defaults to 'RANDOM'.
    """

    objectives: List[str]
    plots_path: Optional[str] = None
    dpi: int = 200
    moo_order: Optional[List[str]] = None
    comparison_columns: Optional[List[str]] = None
    comparison_col: Optional[str] = None
    smiles_col: str = "SMILES"
    include_random: bool = False
    random_label: str = "RANDOM"