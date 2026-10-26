import os
from typing import Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare

from .config import PlotConfig


class MOOPlots:
    """
    Visualization class for multi-objective optimization (MOO) analysis.

    This class provides a comprehensive suite of plotting methods to analyze and compare
    the performance of different MOO algorithms. It generates publication-quality
    visualizations for:
    - Molecule validity and quality metrics
    - Chemical space exploration (diversity, uniqueness)
    - Pareto front analysis
    - Statistical comparisons across methods
    - Performance metrics over optimization steps

    The class supports both saving plots to disk and returning matplotlib Figure objects
    for interactive use, determined by whether plots_path is set.

    Attributes:
        plots_path (Optional[str]): Directory path for saving plots. If None, methods
            return Figure objects instead of saving to disk. Directory is created
            automatically if it doesn't exist.
        dpi (int): Resolution in dots per inch for saved plots. Typical values:
            72 (screen), 150 (draft), 300 (publication quality).
        moo_order (Optional[List[str]]): Ordered list of MOO method names for consistent
            plotting order and color assignment across all visualizations. If None,
            default pandas ordering is used.
        objectives (List[str]): Column names representing optimization objectives
            (e.g., ['QED', 'SA_Score', 'Potency']).
        comparison_columns (List[str]): Hierarchical grouping columns for data analysis,
            typically ['Scaffold', 'MOO', 'Replicate']. Used to aggregate metrics
            across experimental conditions.
        comparison_col (str): Primary comparison column for statistical tests and
            main plot grouping. Defaults to the first entry in comparison_columns.
        smiles_col (str): Name of the column containing SMILES molecular representations.
        include_random (bool): Whether to include a random baseline in comparative plots.
            When True, adds horizontal reference lines showing random performance.
        random_label (str): Label identifying random baseline data in the comparison_col.
            Default is typically 'Random'.

    Example:
        >>> config = PlotConfig(
        ...     plots_path='./analysis/plots',
        ...     objectives=['QED', 'SA_Score'],
        ...     moo_order=['AM', 'GM', 'CS']
        ... )
        >>> plotter = MOOPlots(config)
        >>> fig = plotter.plot_invalid_molecules(moo_results_df)
        >>> fig = plotter.plot_evaluation_summary(summary_df, ['Hypervolume', 'HitRate'])

    Note:
        - All plot methods follow a consistent pattern: they accept a DataFrame and
          optional parameters, and return Optional[plt.Figure].
        - When plots_path is None, figures remain open for further customization.
        - When plots_path is set, figures are automatically saved and closed.
        - Statistical tests (Friedman test) are performed automatically in summary plots.
    """

    def __init__(self, config: PlotConfig) -> None:
        """
        Initialize the MOOPlots instance with configuration settings.

        Args:
            config (PlotConfig): Configuration object containing all plot settings.
                Must include at minimum: objectives list. Other settings have defaults:
                - plots_path: None (returns figures)
                - dpi: 150
                - moo_order: None (uses natural ordering)
                - comparison_columns: ['MOO']
                - smiles_col: 'SMILES'
                - include_random: False

        Raises:
            AttributeError: If required config attributes are missing.
        """
        self.plots_path = config.plots_path
        self.dpi = config.dpi
        self.moo_order = config.moo_order
        self.objectives = config.objectives
        self.comparison_columns = config.comparison_columns or ["MOO"]
        self.comparison_col = config.comparison_col or self.comparison_columns[0]
        self.smiles_col = config.smiles_col
        self.include_random = config.include_random
        self.random_label = config.random_label

    def _save_or_return_figure(self, filename: str) -> Optional[plt.Figure]:
        """
        Save figure to disk or return it for interactive use.

        This helper method implements the dual-mode behavior of the plotting class:
        either persisting figures to disk or returning them for further manipulation.

        Args:
            filename (str): Name of the file to save (e.g., 'InvalidMolecules.jpg').
                Path is automatically prepended from self.plots_path.

        Returns:
            Optional[plt.Figure]:
                - None if plots_path is set (figure saved and closed)
                - Current matplotlib Figure object if plots_path is None

        Side Effects:
            - Creates plots_path directory if it doesn't exist (when saving)
            - Closes the current matplotlib figure (when saving)
            - Saves figure with specified DPI (when saving)
        """
        if self.plots_path is not None:
            os.makedirs(self.plots_path, exist_ok=True)
            plt.savefig(os.path.join(self.plots_path, filename), dpi=self.dpi)
            plt.close()
            return None
        else:
            return plt.gcf()

    def plot_invalid_molecules(
        self, moo_dataframe: pd.DataFrame
    ) -> Optional[plt.Figure]:
        """
        Plot distribution of invalid molecules by MOO method.

        Creates a box-and-swarm plot showing the proportion of chemically invalid
        molecules (e.g., invalid SMILES, unstable structures) generated by each MOO
        algorithm across multiple replicates. Lower proportions indicate better
        chemical validity.

        Args:
            moo_dataframe (pd.DataFrame): MOO results DataFrame with required columns:
                - 'Invalid' (bool): True if molecule is chemically invalid
                - All columns in self.comparison_columns (e.g., 'MOO', 'Scaffold', 'Replicate')

        Returns:
            Optional[plt.Figure]:
                - Figure object if plots_path is None
                - None if figure saved to 'InvalidMolecules.jpg'

        Example:
            >>> df = pd.DataFrame({
            ...     'MOO': ['AM', 'AM', 'GM'],
            ...     'Replicate': [1, 2, 1],
            ...     'Invalid': [False, True, False]
            ... })
            >>> fig = plotter.plot_invalid_molecules(df)

        Note:
            The swarm plot shows individual replicate values, while the box plot
            summarizes the distribution across replicates for each method.
        """
        fig = plt.figure()
        
        # Calculate proportion of invalid molecules per group
        invalid_df = pd.DataFrame(
            moo_dataframe.groupby(self.comparison_columns).value_counts(
                ["Invalid"], normalize=True
            )
        ).reset_index()

        # Box plot for distribution
        sns.boxplot(
            invalid_df[invalid_df["Invalid"]],
            y=0,
            x="MOO",
            fliersize=1,
            order=self.moo_order,
        )
        # Swarm plot for individual replicates
        sns.swarmplot(
            invalid_df[invalid_df["Invalid"]],
            y=0,
            x="MOO",
            alpha=0.9,
            color="darkgray",
            s=3,
            order=self.moo_order,
        )
        plt.ylabel("Proportion invalid")
        plt.tight_layout()

        return self._save_or_return_figure("InvalidMolecules.jpg")

    def plot_unique_molecules(
        self, moo_dataframe: pd.DataFrame
    ) -> Optional[plt.Figure]:
        """
        Plot cumulative unique molecules sampled over optimization steps.

        Displays line plots showing how the number of unique molecules grows across
        optimization iterations for each MOO method. Steeper curves indicate better
        chemical space exploration. Each subplot represents a different scaffold.

        Args:
            moo_dataframe (pd.DataFrame): MOO results DataFrame with required columns:
                - 'step' (int): Optimization iteration number
                - self.smiles_col (str): SMILES string for molecule identification
                - All columns in self.comparison_columns

        Returns:
            Optional[plt.Figure]:
                - Figure object with FacetGrid if plots_path is None
                - None if figure saved to 'UniqueMolecules.jpg'

        Example:
            >>> df = pd.DataFrame({
            ...     'MOO': ['AM'] * 100,
            ...     'Scaffold': ['benzene'] * 100,
            ...     'Replicate': [1] * 100,
            ...     'step': range(100),
            ...     'SMILES': [f'C{i}' for i in range(100)]
            ... })
            >>> fig = plotter.plot_unique_molecules(df)

        Note:
            Duplicates are identified by SMILES string. Only the first occurrence
            of each unique molecule is counted toward the cumulative total.
        """
        fig = plt.figure()
        unique_df = pd.DataFrame()

        for cc, df_ in moo_dataframe.groupby(self.comparison_columns):
            # Sort by step for proper cumulative calculation
            df_ = df_.sort_values("step")
            # Mark first occurrence of each unique molecule
            df_["Unique"] = ~df_.duplicated(self.smiles_col, keep="first")
            # Cumulative sum of unique molecules per step
            df_ = df_.groupby("step").sum("Unique")[["Unique"]].cumsum().reset_index()

            # Add back grouping columns
            for nm, c in zip(self.comparison_columns, cc):
                df_[nm] = c
            unique_df = pd.concat([unique_df, df_])

        sns.relplot(
            kind="line",
            data=unique_df,
            x="step",
            y="Unique",
            hue="MOO",
            col="Scaffold",
            col_wrap=5,
            hue_order=self.moo_order,
        )
        plt.tight_layout()

        return self._save_or_return_figure("UniqueMolecules.jpg")

    def plot_nibr_filters(self, moo_dataframe: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Plot proportion of molecules failing NIBR medicinal chemistry filters.

        NIBR filters flag molecules
        with undesirable medicinal chemistry properties (e.g., reactive groups). Lower failure rates indicate better quality.

        Args:
            moo_dataframe (pd.DataFrame): MOO results DataFrame with required columns:
                - 'Filters' (bool): True if molecule passes NIBR filters
                - All columns in self.comparison_columns

        Returns:
            Optional[plt.Figure]:
                - Figure object if plots_path is None
                - None if figure saved to 'NIBRFilters.jpg'

        Example:
            >>> df = pd.DataFrame({
            ...     'MOO': ['AM', 'GM'],
            ...     'Replicate': [1, 1],
            ...     'Filters': [True, False]
            ... })
            >>> fig = plotter.plot_nibr_filters(df)

        Note:
            Only molecules that FAIL filters are plotted. Lower proportions are better.
        """
        fig = plt.figure()
        
        # Calculate filter pass/fail proportions
        filtered_df = pd.DataFrame(
            moo_dataframe.groupby(self.comparison_columns).value_counts(
                ["Filters"], normalize=True
            )
        ).reset_index()
        filtered_df["Filters"] = filtered_df["Filters"].map(
            {True: "Pass", False: "Fail"}
        )

        # Plot only failures
        sns.boxplot(
            filtered_df[filtered_df["Filters"] == "Fail"],
            y=0,
            x="MOO",
            fliersize=1,
            order=self.moo_order,
        )
        sns.swarmplot(
            filtered_df[filtered_df["Filters"] == "Fail"],
            y=0,
            x="MOO",
            alpha=0.9,
            color="darkgray",
            s=3,
            order=self.moo_order,
        )
        plt.ylabel("Proportion failed NIBR filters")
        plt.tight_layout()

        return self._save_or_return_figure("NIBRFilters.jpg")

    def plot_molecule_summary(
        self, moo_dataframe: pd.DataFrame
    ) -> Optional[plt.Figure]:
        """
        Plot progressive filtering summary showing molecule quality cascade.

        Creates a multi-panel plot showing the progressive reduction in molecule count
        through quality filters:
        1. Valid SMILES (chemically valid structures)
        2. Unique molecules (after removing duplicates)
        3. Pass NIBR filters (after removing problematic structures)

        This "funnel" view shows how much of the generated chemical space survives
        each quality check, revealing the efficiency of each MOO method.

        Args:
            moo_dataframe (pd.DataFrame): Complete MOO results DataFrame with:
                - 'Invalid' (bool): True for invalid SMILES
                - 'Filters' (bool): True if passes NIBR filters
                - self.smiles_col (str): SMILES strings
                - 'Scaffold', 'MOO', 'Replicate': Grouping columns

        Returns:
            Optional[plt.Figure]:
                - FacetGrid figure if plots_path is None
                - None if figure saved to 'MoleculeSummary.jpg'

        Example:
            >>> df = pd.DataFrame({
            ...     'Scaffold': ['benzene'] * 1000,
            ...     'MOO': ['AM'] * 1000,
            ...     'Replicate': [1] * 1000,
            ...     'Invalid': [False] * 950 + [True] * 50,
            ...     'Filters': [True] * 900 + [False] * 100,
            ...     'SMILES': [f'C{i}' for i in range(1000)]
            ... })
            >>> fig = plotter.plot_molecule_summary(df)

        Note:
            Each panel represents a different scaffold. Lines show progression
            through the three quality stages for each MOO method.
        """
        fig = plt.figure()
        progression_df = []
        
        # Calculate counts at each filtering stage
        for (scf, moo, repl), df_ in moo_dataframe.groupby(
            ["Scaffold", "MOO", "Replicate"]
        ):
            # Stage 1: Valid SMILES
            valid_smiles = df_[~df_["Invalid"]]
            # Stage 2: Unique molecules
            non_duplicated = valid_smiles[
                ~valid_smiles.duplicated(self.smiles_col, keep="first")
            ]
            # Stage 3: Pass medicinal chemistry filters
            pass_filters = non_duplicated[non_duplicated["Filters"] == True]
            
            progression_df.append(
                [
                    scf,
                    moo,
                    repl,
                    len(valid_smiles),
                    len(non_duplicated),
                    len(pass_filters),
                ]
            )
            
        progression_df = pd.DataFrame(
            progression_df,
            columns=[
                "Scaffold",
                "MOO",
                "Replicate",
                "Valid Smiles",
                "Unique Smiles",
                "Pass NIBR Filter",
            ],
        )
        # Reshape for plotting
        progression_df = progression_df.melt(id_vars=["Scaffold", "MOO", "Replicate"])
        
        g = sns.relplot(
            progression_df,
            x="variable",
            y="value",
            col="Scaffold",
            hue="MOO",
            col_wrap=5,
            kind="line",
            hue_order=self.moo_order,
        )
        g.set_axis_labels(xlabel="")
        plt.tight_layout()
        
        return self._save_or_return_figure("MoleculeSummary.jpg")

    def plot_evaluation_summary(
        self,
        evaluation_summary: pd.DataFrame,
        eval_columns: List[str],
        plot_type: str = "line",
        share_axis: bool = False,
        p_values: List[float] = [0.05, 0.01, 0.001, 0.0001],
    ) -> Optional[plt.Figure]:
        """
        Create comprehensive evaluation summary with statistical comparisons.

        Generates publication-quality plots comparing MOO methods across multiple
        evaluation metrics with automatic statistical testing (Friedman test).
        Supports multiple visualization types to suit different presentation needs.

        Args:
            evaluation_summary (pd.DataFrame): Evaluation metrics DataFrame with:
                - Columns matching self.comparison_columns (grouping variables)
                - Columns matching eval_columns (metrics to plot)
                Each row typically represents one replicate of one method.
            eval_columns (List[str]): Metric column names to visualize. All should
                be numeric and ideally normalized to [0, 1] range for comparability.
            plot_type (str, optional): Visualization style. Options:
                - 'line': Line plot connecting metric values (good for trends)
                - 'radar': Spider/radar plot (good for profile comparison, max ~8 metrics)
                - 'box': Two-row box plots (shared + independent y-axes)
                - 'box_reduced': Single-row box plots with optional shared y-axis
                Defaults to 'line'.
            share_axis (bool, optional): For 'box_reduced' only. If True, all plots
                share y-axis range [0, 1]. If False, each metric uses optimal range.
                Defaults to False.
            p_values (List[float], optional): P-value thresholds for significance
                annotation in Friedman tests. Labels show smallest threshold exceeded.
                Should be in ascending order. Defaults to [0.05, 0.01, 0.001, 0.0001].

        Returns:
            Optional[plt.Figure]:
                - Figure or FacetGrid object if plots_path is None
                - None if figure saved to 'EvaluationSummary.jpg'

        Raises:
            ValueError: If plot_type is not one of: 'line', 'radar', 'box', 'box_reduced'.

        Example:
            >>> summary_df = pd.DataFrame({
            ...     'MOO': ['AM'] * 10 + ['GM'] * 10,
            ...     'Replicate': list(range(10)) * 2,
            ...     'Hypervolume': np.random.rand(20),
            ...     'Diversity': np.random.rand(20)
            ... })
            >>> fig = plotter.plot_evaluation_summary(
            ...     summary_df, 
            ...     ['Hypervolume', 'Diversity'],
            ...     plot_type='box_reduced',
            ...     share_axis=True
            ... )

        Note:
            - 'box_reduced' type automatically excludes random baseline if 
              self.include_random is True, but shows it as a reference line.
            - Statistical tests compare all MOO methods; 'NS' indicates p >= 0.05.
            - Metrics should be "higher is better" for intuitive interpretation.
            - The ↑ symbol in titles indicates "higher is better" direction.
        """
        fig = plt.figure()
        # Identify non-metric columns (grouping variables)
        id_vars = [c for c in evaluation_summary.columns if c not in eval_columns]

        if plot_type == "line":
            # Reshape data for line plot
            evaluation_summary = evaluation_summary.melt(
                id_vars=id_vars, value_vars=eval_columns
            )
            g = sns.lineplot(
                data=evaluation_summary,
                hue="MOO",
                x="variable",
                y="value",
                dashes=False,
                markers=True,
                markersize=8,
                palette=sns.color_palette(),
                hue_order=self.moo_order,
            )
            g.set(xlabel="")
            g.set(ylabel="")
            g.set_ylim(0, 1)

        elif plot_type == "radar":
            from math import pi

            ax = plt.subplot(111, polar=True)
            N = len(eval_columns)
            # Close the polygon by repeating first value
            values = evaluation_summary[eval_columns + eval_columns[:1]].values
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            plt.xticks(angles[:-1], eval_columns, color="grey", size=8)
            for v, c in zip(values, sns.color_palette()):
                ax.plot(angles, v, linewidth=1, linestyle="solid", color=c)

        elif plot_type == "box":
            # Two-row layout: shared axis (top) and independent axis (bottom)
            fig, axs = plt.subplots(
                2,
                len(eval_columns),
                sharex=False,
                figsize=(len(eval_columns) * 5, 2 * 5),
            )
            evaluation_summary = evaluation_summary.melt(
                id_vars=id_vars, value_vars=eval_columns
            )
            
            for i, ec in enumerate(eval_columns):
                es = evaluation_summary[evaluation_summary["variable"] == ec]
                
                # Top row: shared y-axis [0, 1]
                sns.boxplot(
                    data=es,
                    x="MOO",
                    y="value",
                    hue="MOO",
                    legend=False,
                    ax=axs[0, i],
                    order=self.moo_order,
                    palette=sns.color_palette(),
                    width=0.7,
                    fliersize=4,
                )
                # Bottom row: independent y-axis
                sns.boxplot(
                    data=es,
                    x="MOO",
                    y="value",
                    hue="MOO",
                    legend=False,
                    ax=axs[1, i],
                    order=self.moo_order,
                    palette=sns.color_palette(),
                    width=0.7,
                    fliersize=4,
                )

                # Perform Friedman test
                p_val = friedmanchisquare(
                    *[vals.values for moo, vals in es.groupby("MOO")["value"]]
                ).pvalue
                
                # Determine significance level
                resolved = False
                for p in p_values:
                    if p_val < p:
                        p_val = f"p < {p}"
                        resolved = True
                        break
                if not resolved:
                    p_val = "NS"
                    
                axs[0, i].set_title(ec + "\nShared Axis")
                axs[0, i].set_xlabel("")
                axs[0, i].set_ylim(0, 1)
                axs[1, i].set_title(ec + f"\n{p_val}")
                axs[0, i].set_ylabel("")
                axs[1, i].set_ylabel("")

        elif plot_type == "box_reduced":
            # Single-row box plots with optional shared axis
            fig, axs = plt.subplots(
                1, len(eval_columns), sharex=False, figsize=(len(eval_columns) * 5, 5)
            )
            evaluation_summary = evaluation_summary.melt(
                id_vars=id_vars, value_vars=eval_columns
            )
            
            for i, ec in enumerate(eval_columns):
                es = evaluation_summary[evaluation_summary["variable"] == ec]

                # Exclude random baseline from box plots (shown as reference line instead)
                if self.include_random:
                    es = es[es[self.comparison_col] != self.random_label]
                    
                # Skip if all values are NaN
                if len(es) == len(es[es["value"].isna()]):
                    continue
                    
                sns.boxplot(
                    data=es,
                    x=self.comparison_col,
                    y="value",
                    hue=self.comparison_col,
                    legend=False,
                    ax=axs[i],
                    order=self.moo_order,
                    palette=sns.color_palette(),
                    width=0.7,
                    fliersize=4,
                )

                # Perform Friedman test
                p_val = friedmanchisquare(
                    *[
                        vals.values
                        for moo, vals in es.groupby(self.comparison_col)["value"]
                    ]
                ).pvalue
                
                # Determine significance level
                for p in sorted(p_values):
                    if p_val < p:
                        p_val = f"p < {p}"
                        break
                if type(p_val) != str:
                    p_val = "NS"

                # Apply shared or independent y-axis
                if share_axis:
                    axs[i].set_ylim(0, 1)
                    
                # ↑ indicates "higher is better"
                axs[i].set_title(ec + " (\u2191)" + f"\n{p_val}")
                axs[i].set_ylabel("")

                # Add random baseline reference line
                if self.include_random:
                    random_mean = evaluation_summary[
                        (evaluation_summary["variable"] == ec)
                        & (evaluation_summary[self.comparison_col] == self.random_label)
                    ]["value"].mean()
                    axs[i].axhline(
                        random_mean,
                        ls=":",
                        zorder=0,
                        lw=2,
                        color="gray",
                    )
        else:
            raise ValueError(
                f"Plot type '{plot_type}' not implemented. "
                f"Choose from: 'line', 'radar', 'box', 'box_reduced'"
            )
            
        plt.tight_layout()
        return self._save_or_return_figure("EvaluationSummary.jpg")

    def create_comparison_plot(
        self, 
        df: pd.DataFrame, 
        comparison_metric: str,
        hue_col: Optional[str] = None,
        comparison_col: str = "MOO",
    ) -> Generator[sns.FacetGrid, None, None]:
        """
        Generate pairwise scatter plots comparing MOO method performance.

        Creates a series of comparison grids where each MOO method is compared against
        all others for a given metric. Each yield represents one "base" method compared
        to all alternatives. Green-shaded regions indicate where the base method
        outperforms, red regions where alternatives are better.

        This visualization is particularly useful for head-to-head comparisons across
        multiple experimental conditions (e.g., different scaffolds or replicates).

        Args:
            df (pd.DataFrame): Evaluation results DataFrame with:
                - comparison_metric column (metric to compare)
                - All columns in self.comparison_columns
            comparison_metric (str): Name of the metric column to compare
                (e.g., 'Hypervolume', 'HitRate'). Should be numeric.
            hue_col (Optional[str], optional): Column name for color-coding scatter
                points (e.g., 'Scaffold' to see which scaffolds favor which method).
                If None, all points same color. Defaults to None.

        Yields:
            sns.FacetGrid: Seaborn FacetGrid for each base MOO method. Each grid
                contains scatter plots comparing base method (x-axis) vs. each
                alternative method (y-axis) across all experimental conditions.

        Example:
            >>> df = pd.DataFrame({
            ...     'MOO': ['AM', 'GM'] * 50,
            ...     'Scaffold': ['benzene', 'thiophene'] * 25,
            ...     'Replicate': list(range(25)) * 4,
            ...     'Hypervolume': np.random.rand(100)
            ... })
            >>> for grid in plotter.create_comparison_plot(df, 'Hypervolume', 'Scaffold'):
            ...     plt.show()  # or save: grid.savefig(f'comparison_{base_moo}.png')

        Plot Interpretation:
            - Diagonal line: Equal performance
            - Green shaded area (below diagonal): Base method wins
            - Red shaded area (above diagonal): Comparison method wins
            - Green percentage: Fraction of cases where base method is better
            - Red percentage: Fraction of cases where comparison method is better
            - Points scattered along diagonal: Similar performance

        Note:
            - This is a generator function; call list() to generate all plots at once
            - Each FacetGrid has multiple subplots (one per comparison method)
            - Missing values (NaN) are filled with 0 for comparison purposes
            - The function uses outer merge, so all replicates are included even if
              one method has missing data
        """
        # Remove 'MOO' from comparison columns to merge on other grouping variables
        cc = self.comparison_columns.copy()
        cc.remove(comparison_col)
        
        for base_moo in self.moo_order:
            compare_moo = pd.DataFrame()
            comp_base_moo = df[(df[comparison_col] == base_moo)]
            
            # Create pairwise comparisons with all other methods
            for moo in self.moo_order:
                if moo == base_moo:
                    continue
                    
                comp_moo = df[(df[comparison_col] == moo)]
                # Merge on grouping columns (e.g., Scaffold, Replicate)
                comp_moo = pd.merge(comp_base_moo, comp_moo, on=cc, how="outer")
                comp_moo[f"{comparison_col}_x"] = base_moo
                comp_moo[f"{comparison_col}_y"] = moo
                compare_moo = pd.concat([compare_moo, comp_moo])
                
            compare_moo = compare_moo.rename(columns={f"{comparison_col}_y": f"Comparison{comparison_col}"})

            # Fill missing values with 0 (assumes metric is "higher is better")
            compare_moo[comparison_metric + "_x"] = compare_moo[
                comparison_metric + "_x"
            ].fillna(0)
            compare_moo[comparison_metric + "_y"] = compare_moo[
                comparison_metric + "_y"
            ].fillna(0)
            
            # Create FacetGrid of scatter plots
            g = sns.relplot(
                compare_moo.reset_index(),
                kind="scatter",
                x=comparison_metric + "_x",
                y=comparison_metric + "_y",
                col=f"Comparison{comparison_col}",
                col_wrap=5,
                hue=hue_col,
            )
            max_lim = None
            # Annotate each subplot
            for moo, ax in g.axes_dict.items():
                if max_lim is None:
                    max_lim = max([1, ax.get_xlim()[-1], ax.get_ylim()[-1]])
                ax.set_xlim([0 - (0.05*max_lim), max_lim +  (0.05*max_lim)])
                ax.set_ylim([0 - (0.05*max_lim), max_lim + (0.05*max_lim)])
                
                # Diagonal reference line (equal performance)
                ax.plot([0, max_lim], [0, max_lim], color="gray", zorder=1)
                
                # Shaded regions
                x1 = [0, max_lim]
                y1 = [0, max_lim]
                ax.fill_between(x1, y1, color="green", alpha=0.1)  # Base method wins
                ax.fill_betweenx(y1, x1, color="red", alpha=0.1)   # Comparison wins
                
                # Axis labels
                moo_nm = compare_moo[f"{comparison_col}_x"].unique()[0]
                ax.set_xlabel(f"{comparison_metric} {moo_nm}")
                ax.set_ylabel(f"{comparison_metric} Comparison")

                # Calculate win percentages
                subset = compare_moo[compare_moo[f"Comparison{comparison_col}"] == moo]
                
                # Green region: base method better
                pct_better = (
                    subset[comparison_metric + "_x"] > subset[comparison_metric + "_y"]
                ).sum() / len(subset)
                
                # Red region: comparison method better
                pct_worse = (
                    subset[comparison_metric + "_x"] < subset[comparison_metric + "_y"]
                ).sum() / len(subset)
                
                # Get data bounds
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0]
                
                # Find data point density in candidate regions
                x_data = subset[comparison_metric + "_x"].values
                y_data = subset[comparison_metric + "_y"].values
                
                # Candidate positions (corners and edges)
                candidates = [
                    (0.75, 0.25, "green", pct_better),   # Bottom-right (green region)
                    (0.25, 0.85, "red", pct_worse),       # Top-left (red region)
                    (0.35, 0.15, "green", pct_better),    # Bottom-left
                    (0.65, 0.85, "red", pct_worse),       # Top-right
                    (0.8, 0.6, "green", pct_better),       # Top-right
                    (0.15, 0.35, "red", pct_worse),    # Bottom-left
                ]
                
                # best_pos = None
                # min_density = float('inf')
                densities = []
                for x_norm, y_norm, color, pct in candidates:
                    x_pos = xlim[0] + x_norm * x_range
                    y_pos = ylim[0] + y_norm * y_range
                    
                    # Calculate density of points near this position
                    # Use a window of 15% of plot range
                    x_window = 0.15 * x_range
                    y_window = 0.15 * y_range
                    
                    densities.append((
                        (np.abs(x_data - x_pos) < x_window) &
                        (np.abs(y_data - y_pos) < y_window)
                    ).sum())
                    

                prepped_colors = []

                for i in np.argsort(densities):
                    if len(prepped_colors) == 2:
                        break
                    x_norm, y_norm, color, pct = candidates[i]
                    if color in prepped_colors: continue
                    ax.text(
                        x_norm,
                        y_norm,
                        s=f"{pct:.1%}",
                        fontsize="large",
                        fontweight="demi",
                        color=color,
                        alpha=0.7 if color == "red" else 1.0,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.3, edgecolor="white")
                    )
                    prepped_colors.append(color)

                

            yield g
