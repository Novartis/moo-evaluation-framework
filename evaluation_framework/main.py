"""
Multi-Objective Optimization (MOO) Analysis and Visualization Module

This module provides a comprehensive framework for analyzing and visualizing 
multi-objective optimization results in molecular design. It supports:

- Comparative evaluations of different MOO methods
- Multiple visualization types (box plots, radar charts, etc.)

Typical usage example:
    ```python
    from evaluation_framework.main import MOOEvaluation
    from evaluation_framework.config import AnalysisConfig, DataConfig, PlotConfig
    
    # Configure analysis
    config = AnalysisConfig(
        objectives=['property1', 'property2'],
        comparison_columns=['Scaffold', 'MOO', 'Replicate']
    )
    
    # Initialize evaluation
    evaluator = MOOEvaluation(config)
    evaluator.prep_data(data_config)
    evaluator.prep_plots(plot_config)
    
    # Generate summary and visualizations
    summary = evaluator.generate_evaluation_summary()
    evaluator.evaluation_visual(eval_columns=['Hypervolume', 'Novelty'])
    ```
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import MOOData
from .config import AnalysisConfig, DataConfig, PlotConfig
from .visualise import MOOPlots

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

# Constants
DEFAULT_HR_THRESHOLD = 0.5
HR_THRESHOLD_LOW = 0.3
HR_THRESHOLD_HIGH = 0.7
DEFAULT_DPI = 200


class MOOEvaluation:
    """
    High-level orchestrator for multi-objective optimization evaluation and visualization.

    This class coordinates the complete MOO analysis pipeline, including:
    - Data loading and preprocessing
    - Metric calculation
    - Comparative analysis across methods
    - Visualization generation

    The evaluation process is flexible and allows selective metric calculation
    to optimize performance for large datasets.

    Attributes:
        objectives (List[str]): Names of objective properties to optimize
        comparison_columns (List[str]): Column names for grouping comparisons
            (e.g., ['Scaffold', 'MOO', 'Replicate'])
        moo_data (MOOData): Data management instance for metric calculations
        moo_plot (MOOPlots): Visualization instance for plot generation

    Example:
        ```python
        config = AnalysisConfig(
            objectives=['QED', 'SAS'],
            comparison_columns=['Scaffold', 'MOO']
        )
        evaluator = MOOEvaluation(config)
        ```

    Note:
        The class follows a builder pattern where data and plots are prepared
        separately before running analysis methods.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """
        Initialize the MOOEvaluation orchestrator.

        Args:
            config: Configuration object containing objectives, MOO method ordering,
                and comparison grouping columns.

        Raises:
            ValueError: If config contains empty or invalid values.
        """
        if not config.objectives:
            raise ValueError("Configuration must specify at least one objective")
        if not config.comparison_columns:
            raise ValueError("Configuration must specify comparison columns")
            
        self.objectives = config.objectives
        self.comparison_columns = config.comparison_columns

    def prep_data(self, data_config: DataConfig) -> None:
        """
        Load and prepare MOO data from file.

        Performs initial data quality checks including:
        - Identifying invalid molecules
        - Applying NIBR molecular filters
        - Setting up data structures for analysis

        Args:
            data_config: Configuration object specifying data source and column mappings.

        Raises:
            FileNotFoundError: If data file does not exist.
            ValueError: If required columns are missing from data.

        Note:
            This must be called before any analysis methods.
        """
                
        self.moo_data = MOOData(data_config)
        self.moo_data.assign_invalid()
        self.moo_data.apply_nibrfilters()

    def prep_plots(self, plot_config: PlotConfig) -> None:
        """
        Initialize the plotting subsystem.

        Args:
            plot_config: Configuration object containing plot settings such as
                output directory, DPI, figure size, and style preferences.

        Note:
            If plot_config.plots_path is None, figure objects will be returned
            instead of being saved to disk.
        """
        self.moo_plot = MOOPlots(plot_config)

    def apply_tpp(self, hr_threshold: Union[float, List[float], Dict[str, float]]) -> None:
        """
        Apply Target Product Profile (TPP) thresholds to objectives.

        Args:
            hr_threshold: Hit rate threshold(s). Can be:
                - float: Single threshold applied to all objectives
                - List[float]: Separate threshold per objective (must match objectives length)
                - Dict[str, float]: Objective name to threshold mapping

        Raises:
            ValueError: If threshold list length doesn't match objectives count.
        """
        self.moo_data.apply_tpp(hr_threshold)

    def define_relevant_molecules(
        self, relevant_molecules: Optional[pd.Series] = None
    ) -> None:
        """
        Define which molecules are relevant for analysis.

        Args:
            relevant_molecules: Boolean Series indicating relevant molecules.
                Index should match the main dataframe. If None, all valid
                molecules are considered relevant.

        Note:
            This affects downstream calculations of proportions and metrics.
        """
        self.moo_data.define_relevant_molecules(relevant_molecules)

    def define_hitmolecules(self, hit_molecules: Optional[pd.Series] = None) -> None:
        """
        Define which molecules are hits meeting all criteria.

        A hit molecule typically satisfies all TPP thresholds and passes
        quality filters.

        Args:
            hit_molecules: Boolean Series indicating hit molecules.
                Index should match the main dataframe. If None, hits are
                determined automatically based on TPP and filters.

        See Also:
            apply_tpp: For setting TPP thresholds
        """
        self.moo_data.define_hitmolecules(hit_molecules)

    def set_known_molecules(self, known_molecules: List[str]) -> None:
        """
        Define known molecules for novelty analysis.

        Args:
            known_molecules: List of SMILES strings representing known molecules
                (e.g., existing drugs, literature compounds).

        Example:
            ```python
            known = ['CCO', 'c1ccccc1']
            evaluator.set_known_molecules(known)
            ```
        """
        self.moo_data.set_known_molecules(known_molecules)

    def set_known_molecule_properties(
        self,
        known_molecule_properties: Dict[Tuple, Dict[str, float]],
    ) -> None:
        """
        Set properties of known molecules for comparison.

        Args:
            known_molecule_properties: Nested dictionary where outer keys are tuples
                matching comparison_columns order, and values are dictionaries mapping
                objective names to their values.

        Example:
            ```python
            properties = {
                ('Scaffold1', 'Baseline'): {'QED': 0.7, 'SAS': 3.5},
                ('Scaffold2', 'Baseline'): {'QED': 0.6, 'SAS': 4.0}
            }
            evaluator.set_known_molecule_properties(properties)
            ```

        Raises:
            ValueError: If objective names don't match configured objectives.
        """
        self.moo_data.set_known_molecule_properties(known_molecule_properties)

    def _filter_by_comparison(
        self, df: pd.DataFrame, **filter_values: Any
    ) -> pd.DataFrame:
        """
        Filter dataframe by comparison columns using dynamic keyword arguments.

        This is a utility method for consistent filtering across analysis methods.

        Args:
            df: DataFrame to filter.
            **filter_values: Key-value pairs for filtering where keys are column names
                and values are the desired values (e.g., Scaffold=1, MOO='AM').

        Returns:
            Filtered DataFrame with only rows matching all filter conditions.
        """
        mask = pd.Series([True] * len(df), index=df.index)

        for col, value in filter_values.items():
            if col in df.columns:
                mask &= df[col] == value
        return df[mask]

    def _get_metric_value(
        self,
        df: pd.DataFrame,
        column: str,
        default: Any = np.nan,
        **filter_values: Any
    ) -> Any:
        """
        Safely extract a metric value from filtered dataframe.

        Applies filters, then sums the specified column. Returns default if
        no data remains after filtering.

        Args:
            df: DataFrame containing the metric.
            column: Column name to extract.
            default: Value to return if no data found after filtering.
            **filter_values: Key-value pairs for filtering (passed to _filter_by_comparison).

        Returns:
            Sum of column values after filtering, or default if empty.

        Raises:
            KeyError: If column doesn't exist in dataframe.
        """
        filtered = self._filter_by_comparison(df, **filter_values)
        if len(filtered) > 0:
            return filtered[column].sum()

        return default

    def check_metrics(
        self, metric: str, metrics: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a specific metric should be calculated.

        Used to enable selective metric calculation for performance optimization.

        Args:
            metric: Name of the metric to check.
            metrics: List of metrics to calculate. If None, all metrics are calculated.

        Returns:
            True if the metric should be calculated, False otherwise.
        """
        if metrics is None:
            return True
        return metric in metrics

    def generate_evaluation_summary(
        self, metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation summary calculating all requested metrics.

        This is the main analysis method that computes a wide range of quality,
        diversity, and performance metrics for MOO results.

        Available metrics include:
        - Valid Molecules: Proportion passing basic validity checks
        - Unique Molecules: Proportion after deduplication
        - Pass NIBR Filters: Proportion passing drug-likeness filters
        - Match TPP 0.X: Proportion meeting TPP at various thresholds (0.3, 0.5, 0.7)
        - Average TD per step: Chemical diversity per generation step
        - Proportion Pareto Molecules: Fraction on Pareto front
        - Hypervolume Pareto Molecules: Hypervolume indicator
        - Similarity Ideal Vector: Distance to ideal point
        - Internal Tanimoto Distance: Within-method chemical diversity
        - Minimum Novelty: Distance to known molecules
        - Unique MOO Molecules: Molecules unique to each method
        - Nondominated Solutions: Count of nondominated solutions vs. known molecules

        Args:
            metrics: List of specific metric names to calculate. If None, calculates
                all available metrics. Use for performance optimization on large datasets.

        Returns:
            DataFrame indexed by comparison_columns with one column per metric.

        Raises:
            AttributeError: If required data hasn't been prepared (call prep_data first).
            ValueError: If known_molecules not set when calculating novelty metrics.

        Example:
            ```python
            # Calculate all metrics
            summary = evaluator.generate_evaluation_summary()
            
            # Calculate only specific metrics
            summary = evaluator.generate_evaluation_summary(
                metrics=['Hypervolume Pareto Molecules', 'Minimum Novelty']
            )
            ```

        Note:
            Some metrics require additional setup:
            - Novelty metrics: Call set_known_molecules() first
            - Nondominated solutions: Call set_known_molecule_properties() first
        """
        usefulmols_df = self.moo_data.get_useful_molecules()

        # Pre-calculate commonly used DataFrames to avoid redundant computation
        if self.check_metrics("Average TD per step", metrics):
            log.info("Preparing Average TD per step")
            td_steps = (
                self.moo_data.tanimotodistance_step
                if not self.moo_data.tanimotodistance_step.empty
                else self.moo_data.get_chemical_breadth()
            )

        if self.check_metrics("Hypervolume Pareto Molecules", metrics):
            log.info("Preparing Hypervolume Pareto Molecules")
            hypervolume_bestmols = (
                self.moo_data.hypervolume_bestmols
                if not self.moo_data.hypervolume_bestmols.empty
                else self.moo_data.get_hypervolume()
            )

        if self.check_metrics("Internal Tanimoto Distance", metrics):
            log.info("Preparing Internal Tanimoto Distance")
            td_internal = (
                self.moo_data.tanimotodistance_internal
                if not self.moo_data.tanimotodistance_internal.empty
                else self.moo_data.get_internal_tanimoto_distance()
            )

        if self.check_metrics("Minimum Novelty", metrics):
            log.info("Preparing Minimum Novelty")
            if not self.moo_data.tanimotodistance_novelty.empty:
                td_novelty = self.moo_data.tanimotodistance_novelty
            else:
                if hasattr(self.moo_data, "known_molecules"):
                    td_novelty = self.moo_data.get_novelty_tanimoto_distance(
                        self.moo_data.known_molecules
                    )
                else:
                    td_novelty = pd.DataFrame()
                    log.warning(
                        "Known molecules must be defined to calculate novelty metrics. "
                        "Call set_known_molecules() before generating summary."
                    )

        if (self.check_metrics("Proportion Pareto Molecules", metrics) 
            and self.moo_data.pareto_molecules.empty):
            log.info("Preparing Proportion Pareto Molecules")
            self.moo_data.get_pareto_molecules()

        if self.check_metrics("Unique MOO Molecules", metrics):
            log.info("Preparing Unique MOO Molecules")
            df_uniq = self.moo_data.get_unique_moo_molecules()
            df_uniq = df_uniq[df_uniq["Unique"]]
            df_uniq["HitRateUniq"] = df_uniq["HitRateUniqPercent"] / 100
            df_uniq = df_uniq.groupby(self.comparison_columns).mean().reset_index()

        if self.check_metrics("Similarity Ideal Vector", metrics):
            log.info("Preparing Similarity Ideal Vector")
            dist_df = self.moo_data.get_distance_from_ideal(similarity=True)

        if self.check_metrics("Match TPP 0.3", metrics):
            log.info("Preparing Match TPP 0.3")
            self.moo_data.set_hitrate(hr_threshold=HR_THRESHOLD_LOW)
            usefulmols03_df = self.moo_data.get_useful_molecules()

        if self.check_metrics("Match TPP 0.7", metrics):
            log.info("Preparing Match TPP 0.7")
            self.moo_data.set_hitrate(hr_threshold=HR_THRESHOLD_HIGH)
            usefulmols07_df = self.moo_data.get_useful_molecules()

        if self.check_metrics("Nondominated Solutions", metrics):
            log.info("Preparing Nondominated Solutions")
            if hasattr(self.moo_data, "known_molecule_properties"):
                nondominated_solutions = self.moo_data.get_nondominated_solutions()
            else:
                nondominated_solutions = pd.DataFrame()
                log.warning(
                    "Known molecule properties must be defined to calculate "
                    "nondominated solution metrics. Call set_known_molecule_properties() first."
                )

        # Reset to default threshold
        self.moo_data.set_hitrate(hr_threshold=DEFAULT_HR_THRESHOLD)

        # Build summary dictionary
        compound_summary = collections.defaultdict(list)
        
        for group_values, df_ in usefulmols_df.groupby(self.comparison_columns):
            # Build filter dict from comparison_columns and group_values
            if isinstance(group_values, tuple):
                current_filters = dict(zip(self.comparison_columns, group_values))
            else:
                current_filters = {self.comparison_columns[0]: group_values}

            # Add grouping columns to summary
            for col, val in current_filters.items():
                compound_summary[col].append(val)

            # Calculate base quality metrics
            compound_quality_filters = {"Invalid": False}
            
            value = self._get_metric_value(df_, "proportion", **compound_quality_filters)
            compound_summary["Valid Molecules"].append(value)

            compound_quality_filters["Duplicated"] = False
            value = self._get_metric_value(df_, "proportion", **compound_quality_filters)
            compound_summary["Unique Molecules"].append(value)

            compound_quality_filters["Filters"] = True
            value = self._get_metric_value(df_, "proportion", **compound_quality_filters)
            compound_summary["Pass NIBR Filters"].append(value)

            # TPP match at default threshold
            if self.check_metrics("Match TPP 0.5", metrics):
                compound_quality_filters["HitRate"] = True
                value = self._get_metric_value(df_, "proportion", **compound_quality_filters)
                compound_summary["Match TPP 0.5"].append(value)

            # Diversity and performance metrics
            if self.check_metrics("Average TD per step", metrics):
                if not self.moo_data.tanimotodistance_step.empty:
                    value = self._get_metric_value(td_steps, "PrevStepTD", **current_filters)
                    compound_summary["Average TD per step"].append(value)

            if self.check_metrics("Proportion Pareto Molecules", metrics):
                if not self.moo_data.pareto_molecules.empty:
                    value = self._get_metric_value(
                        self.moo_data.pareto_molecules, "proportion", **current_filters
                    )
                    compound_summary["Proportion Pareto Molecules"].append(value)

            if self.check_metrics("Hypervolume Pareto Molecules", metrics):
                if not self.moo_data.hypervolume_bestmols.empty:
                    value = self._get_metric_value(
                        hypervolume_bestmols, "Hypervolume", **current_filters
                    )
                    compound_summary["Hypervolume Pareto Molecules"].append(value)

            if self.check_metrics("Similarity Ideal Vector", metrics):
                if not dist_df.empty:
                    value = self._get_metric_value(dist_df, "Euclidean", **current_filters)
                    compound_summary["Similarity Ideal Vector"].append(value)

            if self.check_metrics("Internal Tanimoto Distance", metrics):
                if not td_internal.empty:
                    value = self._get_metric_value(
                        td_internal, "TanimotoDistance", **current_filters
                    )
                    compound_summary["Internal Tanimoto Distance"].append(value)

            if self.check_metrics("Minimum Novelty", metrics):
                if not td_novelty.empty:
                    value = self._get_metric_value(
                        td_novelty, "KnownMoleculesTanimotoDistance", **current_filters
                    )
                    compound_summary["Minimum Novelty"].append(value)

            if self.check_metrics("Unique MOO Molecules", metrics):
                value = self._get_metric_value(df_uniq, "HitRateUniq", **current_filters)
                compound_summary["Unique MOO Molecules"].append(value)

            if self.check_metrics("Nondominated Solutions", metrics):
                if not nondominated_solutions.empty:
                    value = self._get_metric_value(
                        nondominated_solutions, "Nondominated Solutions", **current_filters
                    )
                    compound_summary["Nondominated Solutions"].append(value)

            # TPP matches at alternative thresholds
            if self.check_metrics("Match TPP 0.3", metrics):
                compound_quality_filters["HitRate"] = True
                filters_with_hr = {**current_filters, **compound_quality_filters}
                value = self._get_metric_value(usefulmols03_df, "proportion", **filters_with_hr)
                compound_summary["Match TPP 0.3"].append(value)

            if self.check_metrics("Match TPP 0.7", metrics):
                compound_quality_filters["HitRate"] = True
                filters_with_hr = {**current_filters, **compound_quality_filters}
                value = self._get_metric_value(usefulmols07_df, "proportion", **filters_with_hr)
                compound_summary["Match TPP 0.7"].append(value)

        compound_summary = pd.DataFrame.from_dict(compound_summary).set_index(
            self.comparison_columns
        )
        return compound_summary

    def evaluation_visual(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        eval_columns: Optional[List[str]] = None,
        plot_type: str = "box",
        **kwargs: Any,
    ) -> Optional[plt.Figure]:
        """
        Generate evaluation summary visualizations.

        Creates various plot types to visualize MOO performance metrics across
        methods, scaffolds, or other comparison dimensions.

        Args:
            dataframe: Pre-computed evaluation summary DataFrame. If None, generates
                summary automatically using generate_evaluation_summary().
            eval_columns: Specific metric columns to plot. If None with auto-generated
                dataframe, plots all available metrics.
            plot_type: Type of visualization to generate. Options:
                - 'box': Box plot showing distributions across replicates
                - 'line': Line plot for trend analysis
                - 'radar': Radar/spider chart for multi-metric comparison
                - 'box_reduced': Compact box plot with fewer annotations
            **kwargs: Additional arguments passed to the plotting function, such as:
                - figsize: Tuple for figure dimensions
                - color_palette: Color scheme for plots
                - title: Custom plot title

        Returns:
            matplotlib Figure object if plots_path is None in config, otherwise None
            (plots are saved to disk).

        Raises:
            ValueError: If eval_columns specified but not found in dataframe.
            AssertionError: If dataframe provided but eval_columns is None.

        Example:
            ```python
            # Auto-generate summary and plot
            fig = evaluator.evaluation_visual(
                eval_columns=['Match TPP 0.3', 'Match TPP 0.5', 'Match TPP 0.7'],
                plot_type='radar'
            )
            
            # Use pre-computed summary
            summary = evaluator.generate_evaluation_summary()
            fig = evaluator.evaluation_visual(
                dataframe=summary,
                eval_columns=['Match TPP 0.5', 'Match TPP 0.7'],
                plot_type='reduced_box'
            )
            ```
        """
        if dataframe is None:
            eval_summ_df = self.generate_evaluation_summary(metrics=eval_columns)
            if eval_columns is None:
                eval_columns = eval_summ_df.columns.tolist()
            eval_summ_df = eval_summ_df.reset_index()
            return self.moo_plot.plot_evaluation_summary(
                eval_summ_df, eval_columns, plot_type, **kwargs
            )
        else:
            if eval_columns is None:
                raise AssertionError(
                    "eval_columns must be specified when providing a dataframe"
                )
            if not all(col in dataframe.columns for col in eval_columns):
                missing = [col for col in eval_columns if col not in dataframe.columns]
                raise ValueError(
                    f"Dataframe must contain all eval_columns. Missing: {missing}"
                )
            return self.moo_plot.plot_evaluation_summary(
                dataframe, eval_columns, plot_type, **kwargs
            )
