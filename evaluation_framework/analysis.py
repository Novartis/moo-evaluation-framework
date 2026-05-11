import logging
from functools import cache, lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pygmo as pg
from pygmo import hypervolume
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
from scipy.spatial import distance

from .config import DataConfig
from .nibrfilters import assignFilters, buildFilterCatalog
from .utils import check_strict_filters

log = logging.getLogger(__name__)


class MOOData:
    """
    Data management and analysis class for multi-objective optimization results.

    Handles loading, processing, and calculating various metrics on MOO data including
    Pareto fronts, chemical diversity, hit rates, and quality assessments.

    Attributes:
        moo_dataframe (pd.DataFrame): Main data containing MOO results
        objectives (List[str]): List of objective property names
        smiles_col (str): Column name for SMILES strings
        comparison_columns (List[str]): Columns used for grouping analyses
        ran_duplicated (bool): Flag indicating if duplicate analysis has been run
        tanimotodistance_step (pd.DataFrame): Step-wise Tanimoto distance data
        tanimotodistance_internal (pd.DataFrame): Internal diversity Tanimoto distance data
        tanimotodistance_novelty (pd.DataFrame): Novelty Tanimoto distance data
        pareto_molecules (pd.DataFrame): Pareto optimal molecule data
        hypervolume_bestmols (pd.DataFrame): Hypervolume metric data
        relevant_molecules (pd.Series): Boolean mask for relevant molecules
        hit_molecules (pd.Series): Boolean mask for hit molecules
    """

    def __init__(
        self,
        config: DataConfig,
    ) -> None:
        """
        Initialize MOOData from CSV file.

        Args:
            config (DataConfig): Configuration object containing data path, column names,
                and objective properties. Should have attributes:
                - data_path: Path to CSV file containing MOO results
                - smiles_col: Name of column containing SMILES strings
                - objectives: List of objective property column names
                - comparison_columns: Columns for grouping comparisons (e.g., ['Scaffold', 'MOO', 'Replicate'])
                - invalid_str: Prefix string for invalid molecules
                - replicate_col: Column name for replicates
                - moo_col: Column name for MOO method identifier
                - epoch_col: Column name for optimization epochs
                - score_col: Column name for scores
                - load_all_columns: Whether to load all columns or only specified ones

        Raises:
            ValueError: If data format is not CSV
        """
        self.objectives = config.objectives
        self.smiles_col = config.smiles_col
        self.comparison_columns = config.comparison_columns or ["_"]
        self.invalid_str = config.invalid_str
        self.replicate_col = config.replicate_col
        self.moo_col = config.moo_col
        self.epoch_col = config.epoch_col
        self.score_col = config.score_col

        self.ran_duplicated = False
        self.tanimotodistance_step = pd.DataFrame()
        self.tanimotodistance_internal = pd.DataFrame()
        self.tanimotodistance_novelty = pd.DataFrame()
        self.pareto_molecules = pd.DataFrame()
        self.hypervolume_bestmols = pd.DataFrame()
        # self.relevant_molecules: Optional[pd.Series] = None
        # self.hit_molecules: Optional[pd.Series] = None
        # self.known_molecules: Optional[List[str]] = None
        # self.known_molecule_properties: Optional[Dict] = None
        # self.nondominated_solutions = pd.DataFrame()

        if config.data_path.endswith(".csv"):
            if not config.load_all_columns:
                columns_to_load = (
                    [self.smiles_col]
                    + self.objectives
                    + (
                        self.comparison_columns
                        if self.comparison_columns != ["_"]
                        else []
                    )
                    + ["NIBRFilters"]
                )
                optional_cols = [
                    self.replicate_col,
                    self.moo_col,
                    self.epoch_col,
                    self.score_col,
                ]

                optional_cols = [col for col in optional_cols if col]
                columns_to_load.extend(optional_cols)

                if len(optional_cols) < 3:
                    log.warning(
                        "Optional columns are missing from the data load. This may affect analysis. Please ensure replicate_col, moo_col, and epoch_col are set correctly if relevant."
                    )

                self.moo_dataframe = pd.read_csv(
                    config.data_path,
                    usecols=lambda c: c in columns_to_load,
                )
            else:
                self.moo_dataframe = pd.read_csv(config.data_path)
        else:
            raise ValueError("Only CSV data format is currently supported.")

        if config.comparison_columns is None:
            self.moo_dataframe[self.comparison_columns] = 1

    def clear(self) -> None:
        """
        Reset all cached calculations and metrics.

        Clears all analysis results, allowing for fresh calculation with different parameters.
        """
        self.ran_duplicated = False
        self.tanimotodistance_step = pd.DataFrame()
        self.tanimotodistance_internal = pd.DataFrame()
        self.tanimotodistance_novelty = pd.DataFrame()
        self.pareto_molecules = pd.DataFrame()
        self.hypervolume_bestmols = pd.DataFrame()

    def _average_across_comparison_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average dataframe values across comparison columns.

        Computes mean values grouped by comparison columns and resets the index.

        Args:
            df (pd.DataFrame): DataFrame to average. Should contain columns matching
                self.comparison_columns and numeric columns to average.

        Returns:
            pd.DataFrame: Averaged DataFrame with comparison columns and mean values
        """
        return df.groupby(self.comparison_columns).mean().reset_index()

    def set_hitrate(self, hr_threshold: Union[float, List[float]] = 0.5) -> None:
        """
        Calculate and set HitRate column based on threshold.

        Marks molecules as hits if all objectives meet or exceed the specified threshold(s).
        Sets a 'HitRate' column in moo_dataframe as boolean.

        Args:
            hr_threshold (Union[float, List[float]]): Threshold value(s) for objectives.
                If float: same threshold applied to all objectives.
                If List[float]: one threshold per objective (must match objectives order).

        Returns:
            None
        """
        self.moo_dataframe["HitRate"] = (
            self.moo_dataframe[self.objectives].fillna(1) >= hr_threshold
        ).all(axis=1)

    def apply_tpp(
        self, hr_threshold: Union[float, List[float], Dict[str, float]]
    ) -> None:
        """
        Apply Target Product Profile (TPP) thresholds.

        Wrapper for set_hitrate that accepts multiple threshold formats including
        dictionaries mapping objective names to threshold values.

        Args:
            hr_threshold (Union[float, List[float], Dict[str, float]]): Threshold value(s).
                - float: Same threshold for all objectives
                - List[float]: One threshold per objective (must match objectives order)
                - Dict[str, float]: Maps objective names to their individual thresholds

        Returns:
            None

        Raises:
            KeyError: If dict keys don't match objective names
        """
        if isinstance(hr_threshold, list):
            log.warning(
                "Using list as hr threshold, please ensure order of thresholds matches objectives variable"
            )
        elif isinstance(hr_threshold, dict):
            hr_threshold = [hr_threshold[o] for o in self.objectives]
        self.set_hitrate(hr_threshold)

    def calculate_hitrate(
        self, hr_threshold: Union[float, List[float]] = 0.5
    ) -> pd.Series:
        """
        Calculate hit rate without modifying the dataframe.

        Returns a boolean Series indicating which molecules meet hit criteria without
        permanently modifying the moo_dataframe. Useful for testing different thresholds.

        Args:
            hr_threshold (Union[float, List[float]]): Threshold value(s) for objectives.
                If float: same threshold applied to all objectives.
                If List[float]: one threshold per objective (must match objectives order).

        Returns:
            pd.Series: Boolean mask where True indicates molecules meeting hit criteria
        """
        return (self.moo_dataframe[self.objectives].fillna(1) >= hr_threshold).all(
            axis=1
        )

    def _process_invalid(self, smiles: str) -> bool:
        """
        Check if a SMILES string is invalid.

        Attempts to parse the SMILES string using RDKit's MolFromSmiles.
        Any exception during parsing is interpreted as an invalid SMILES string.

        Args:
            smiles (str): SMILES string to validate

        Returns:
            bool: True if SMILES is invalid, False if valid
        """
        return Chem.MolFromSmiles(smiles) is None

    def assign_invalid(self) -> None:
        """
        Mark invalid molecules as Invalid column in moo_dataframe.

        Identifies invalid molecules through two methods:
        1. Prefix matching: SMILES starting with invalid_str are marked invalid
        2. RDKit parsing: Attempts MolFromSmiles parsing for remaining molecules

        Sets 'Invalid' boolean column in moo_dataframe.

        Returns:
            None
        """
        self.moo_dataframe["Invalid"] = False

        if self.invalid_str:
            self.moo_dataframe["Invalid"] = self.moo_dataframe[
                self.smiles_col
            ].str.startswith(self.invalid_str)

        self.moo_dataframe.loc[~self.moo_dataframe["Invalid"], "Invalid"] = (
            self.moo_dataframe.loc[~self.moo_dataframe["Invalid"], self.smiles_col].map(
                self._process_invalid
            )
        )

    def apply_nibrfilters(self) -> None:
        """
        Apply NIBR medicinal chemistry filters to molecules.

        Calculates and assigns filter results to 'Filters' column if not already present.
        For large datasets (>100000 rows), prompts user to run filtering separately.
        Processes molecules in chunks for efficiency.

        The NIBRFilters are parsed and passed to check_strict_filters to generate
        the final 'Filters' boolean column.

        Returns:
            None

        Raises:
            AssertionError: If NIBRFilters column cannot be created or loaded

        Warnings:
            Logs warning if dataframe exceeds 100000 rows
            Logs warning if required optional columns are missing
        """
        if "NIBRFilters" not in self.moo_dataframe:
            if len(self.moo_dataframe) > 100000:
                log.warning(
                    "Dataframe too large - please run the NIBR Filters pipeline separately"
                )
            else:
                smiles = self.moo_dataframe[~self.moo_dataframe["Invalid"]][
                    self.smiles_col
                ].unique()

                # Determine chunk size for processing
                chunk_size = 1
                for x in range(1, 50000):
                    if len(smiles) % x == 0:
                        chunk_size = x
                        break

                log.info(f"Using chunk size: {chunk_size}")

                catalog = buildFilterCatalog()
                nibr_scores = []

                for chunk in np.split(
                    list(smiles), np.arange(chunk_size, len(smiles), chunk_size)
                ):
                    nibr_scores.extend(
                        list(map(lambda x: assignFilters(catalog, x), chunk))
                    )

                nibr_scores_smiles = {
                    smi: str(list(scr)) for smi, scr in zip(smiles, nibr_scores)
                }
                self.moo_dataframe["NIBRFilters"] = self.moo_dataframe[
                    self.smiles_col
                ].map(nibr_scores_smiles)

        assert "NIBRFilters" in self.moo_dataframe, (
            "Please run the NIBR Filters pipeline, dataframe should be named 'NIBRFilters'"
        )

        self.moo_dataframe["NIBRFilters"] = self.moo_dataframe["NIBRFilters"].apply(
            lambda x: eval(x) if isinstance(x, str) else np.nan
        )
        self.moo_dataframe["Filters"] = self.moo_dataframe["NIBRFilters"].apply(
            check_strict_filters
        )

    def calculate_duplicated(self) -> None:
        """
        Identify and mark duplicate molecules.

        Marks molecules as duplicated if they appear multiple times within the same
        comparison group. Sets 'Duplicated' boolean column and ran_duplicated flag.

        Returns:
            None
        """
        self.moo_dataframe["Duplicated"] = self.moo_dataframe.duplicated(
            [self.smiles_col] + self.comparison_columns
        )
        self.ran_duplicated = True

    def get_useful_molecules(self) -> pd.DataFrame:
        """
        Calculate proportions of molecules by quality category.

        Computes proportions of molecules across combinations of quality metrics:
        Invalid, Duplicated, Filters, and HitRate status. Runs calculate_duplicated
        if not already executed.

        Returns:
            pd.DataFrame: Multi-index DataFrame with columns:
                - comparison_columns: Grouping columns
                - Invalid: Boolean, molecule validity
                - Duplicated: Boolean, duplication status
                - Filters: Boolean, filter pass status
                - HitRate: Boolean, hit status
                - proportion: Float, proportion of molecules in category
        """
        if not self.ran_duplicated:
            self.calculate_duplicated()

        usefulmols_df = pd.DataFrame(
            self.moo_dataframe.groupby(self.comparison_columns).value_counts(
                ["Invalid", "Duplicated", "Filters", "HitRate"], normalize=True
            )
        ).reset_index()
        usefulmols_df = usefulmols_df.rename(columns={0: "proportion"})
        return usefulmols_df

    def define_relevant_molecules(
        self, relevant_molecules: Optional[pd.Series] = None
    ) -> None:
        """
        Define which molecules are relevant for analysis.

        Stores a boolean mask in self.relevant_molecules to identify molecules
        suitable for downstream analysis. Default includes valid, unique, and
        filter-passing molecules.

        Args:
            relevant_molecules (Optional[pd.Series]): Boolean mask; if None, uses
                valid, unique, filter-passing molecules within each comparison group.
                Default: (~Invalid) & (~Duplicated) & (Filters)

        Returns:
            None
        """
        if relevant_molecules is None:
            relevant_molecules = (
                (~self.moo_dataframe["Invalid"])
                & (
                    ~self.moo_dataframe.duplicated(
                        [self.smiles_col] + self.comparison_columns
                    )
                )
                & (self.moo_dataframe["Filters"])
            )
        self.relevant_molecules = relevant_molecules

    def define_hitmolecules(self, hit_molecules: Optional[pd.Series] = None) -> None:
        """
        Define which molecules are hits (meeting all criteria).

        Stores a boolean mask in self.hit_molecules identifying molecules meeting
        all quality and activity criteria. Default includes hits passing filters
        and quality checks.

        Args:
            hit_molecules (Optional[pd.Series]): Boolean mask; if None, uses molecules
                meeting HitRate criteria and quality filters.
                Default: (HitRate) & (~Invalid) & (~Duplicated) & (Filters)

        Returns:
            None
        """
        if hit_molecules is None:
            hit_molecules = (
                (self.moo_dataframe["HitRate"])
                & (~self.moo_dataframe["Invalid"])
                & (
                    ~self.moo_dataframe.duplicated(
                        [self.smiles_col] + self.comparison_columns
                    )
                )
                & (self.moo_dataframe["Filters"])
            )
        self.hit_molecules = hit_molecules

    @lru_cache(maxsize=100000)
    def _get_fingerprint(self, smiles: str) -> Optional[object]:
        """
        Internal method to calculate Morgan fingerprint for a SMILES string.

        Uses RDKit to generate 2048-bit Morgan fingerprints with radius 2.
        Results are cached for efficiency.

        Args:
            smiles (str): SMILES string to fingerprint

        Returns:
            Optional[object]: RDKit fingerprint object if valid, None if invalid SMILES

        Note:
            Results are cached; same SMILES always returns same fingerprint object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return fp
        else:
            return None

    @cache
    def _calculate_fingerprints(self, smiles_list: Tuple[str]) -> Tuple:
        """
        Internal method to calculate Morgan fingerprints for a tuple of SMILES.

        Generates fingerprints for a set of SMILES strings. Uses _get_fingerprint
        internally, so individual fingerprints are cached.

        Args:
            smiles_list (Tuple[str]): Immutable tuple of SMILES strings

        Returns:
            Tuple: Tuple of RDKit fingerprint objects (None for invalid SMILES)

        Note:
            Takes tuple (not list) to enable caching. Results are fully cached.
        """
        fps = [self._get_fingerprint(smi) for smi in smiles_list]
        return tuple(fps)

    def get_distance_from_ideal(self, similarity: bool = False) -> pd.DataFrame:
        """
        Calculate Euclidean distance from ideal objective vector.

        Computes distance from ideal objective vector (all ones) for relevant molecules.
        Groups by comparison_columns and returns median distance per group.

        Args:
            similarity (bool): If True, returns similarity score (1 - normalized distance);
                if False, returns raw Euclidean distance. Default: False

        Returns:
            pd.DataFrame: Median distance/similarity per comparison group, columns:
                - comparison_columns: Grouping columns
                - Euclidean: Distance (or similarity if similarity=True)

        Note:
            Uses relevant_molecules mask; run define_relevant_molecules first
        """
        moo_dataframe = self.moo_dataframe[self.relevant_molecules]
        dist = []
        max_distance = distance.euclidean(
            np.zeros(len(self.objectives)), np.ones(len(self.objectives))
        )
        for group_cols, df in moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )
            for a in df[self.objectives].values:
                value = distance.euclidean(a, np.ones(a.shape))
                if not similarity:
                    dist.append(group_cols + [value])
                else:
                    dist.append(group_cols + [(max_distance - value) / max_distance])
        dist_df = pd.DataFrame(dist, columns=self.comparison_columns + ["Euclidean"])
        dist_df = dist_df.groupby(self.comparison_columns).quantile(0.5).reset_index()
        return dist_df

    def _calculate_tanimoto_distance(
        self, fps1: List, fps2: Optional[List] = None
    ) -> Union[np.ndarray, float]:
        """
        Internal method to calculate Tanimoto distances between fingerprint sets.

        Computes pairwise Tanimoto distances (1 - Tanimoto similarity) between
        fingerprints. If fps2 is None, computes internal pairwise distances.

        Args:
            fps1 (List): List of RDKit fingerprint objects
            fps2 (Optional[List]): List of fingerprints to compare against fps1.
                If None, computes internal pairwise distances. Default: None

        Returns:
            Union[np.ndarray, float]: 
                - If fps2 is provided: scalar mean Tanimoto distance
                - If fps2 is None: array of mean Tanimoto distances per fingerprint in fps1
        """
        if fps2 is not None:
            bts = [
                np.mean(1 - np.array(cDataStructs.BulkTanimotoSimilarity(fp, fps1)))
                for fp in fps2
            ]
        else:
            bts = np.array(
                [
                    1 - np.array(cDataStructs.BulkTanimotoSimilarity(fp, fps1))
                    for fp in fps1
                ]
            )
            np.fill_diagonal(bts, np.nan)
            bts = [np.nanmean(t) for t in bts]
            return np.array(bts)
        if len(bts) > 0:
            return np.mean(bts)
        else:
            return np.nan

    def _calculate_stepwise_tanimoto(
        self, df: pd.DataFrame, stp1: int, stp2: int
    ) -> Union[np.array, float]:
        """
        Internal method to calculate Tanimoto distance between molecules at two epochs.

        Filters molecules by epoch_col and computes distance between the two sets.

        Args:
            df (pd.DataFrame): DataFrame with 'Molecule' column (fingerprints) and epoch_col
            stp1 (int): First epoch number
            stp2 (int): Second epoch number

        Returns:
            Union[np.array, float]: Mean Tanimoto distance between epochs, np.nan if insufficient data
        """
        return self._calculate_tanimoto_distance(
            df[df[self.epoch_col] == stp1]["Molecule"].tolist(),
            df[df[self.epoch_col] == stp2]["Molecule"].tolist(),
        )

    def get_chemical_breadth(self, nsteps: int = 10) -> pd.DataFrame:
        """
        Calculate chemical diversity progression over optimization steps.

        Computes Tanimoto distances between molecules at different steps to measure
        how chemical space explored evolves during optimization. Calculates both
        distance from previous nsteps and distance from first sample.

        Args:
            nsteps (int): Number of steps back to compare for progression. Default: 10

        Returns:
            pd.DataFrame: Chemical breadth metrics per comparison group, columns:
                - comparison_columns: Grouping columns
                - epoch_col: Epoch number
                - PrevStepTD: Tanimoto distance from nsteps back
                - FirstSampleTD: Tanimoto distance from first sample

        Note:
            Excludes invalid molecules. Requires epoch_col to be set in config.
        """

        for group_cols, df in self.moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )

            # df_ = df_.drop(df_[~df_["Invalid"]].index)
            df = df.loc[~df["Invalid"]].copy()
            df["Molecule"] = self._calculate_fingerprints(
                tuple(df[self.smiles_col].tolist())
            )

            bts = [
                self._calculate_stepwise_tanimoto(df, stp - nsteps, stp)
                for stp in range(
                    df[self.epoch_col].min() + nsteps, df[self.epoch_col].max() + 1
                )
            ]
            bts_first = [
                self._calculate_stepwise_tanimoto(df, 1, stp)
                for stp in range(
                    df[self.epoch_col].min() + 1, df[self.epoch_col].max() + 1
                )
            ]

            steps = range(
                df[self.epoch_col].min() + nsteps, df[self.epoch_col].max() + 1
            )
            new_rows = [
                group_cols + [stp, prev_td, first_td]
                for stp, prev_td, first_td in zip(steps, bts, bts_first)
            ]
            chemical_breadth = pd.DataFrame(
                new_rows,
                columns=self.comparison_columns
                + [self.epoch_col, "PrevStepTD", "FirstSampleTD"],
            )

            self.tanimotodistance_step = pd.concat(
                [
                    self.tanimotodistance_step,
                    chemical_breadth.groupby(self.comparison_columns)
                    .mean()
                    .reset_index(),  # TODO: this is overcomplicated - cleanup
                ]
            ).reset_index(drop=True)
        return self.tanimotodistance_step

    def get_internal_tanimoto_distance(self) -> pd.DataFrame:
        """
        Calculate average Tanimoto distance between all pairs of hit molecules.

        Measures internal chemical diversity within the hit set for each MOO method.
        Computes pairwise distances and returns the average per comparison group.

        Returns:
            pd.DataFrame: Average pairwise Tanimoto distances, columns:
                - comparison_columns: Grouping columns
                - TanimotoDistance: Mean pairwise Tanimoto distance (np.nan if ≤1 molecule)

        Note:
            Uses hit_molecules mask; run define_hitmolecules first.
            Final result is averaged across comparison_columns.
        """
        moo_dataframe = self.moo_dataframe[self.hit_molecules]
        tanimoto_distance = pd.DataFrame()
        for group_cols, df in moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )
            smiles = df[self.smiles_col].tolist()
            if len(smiles) <= 1:
                td = [group_cols + [np.nan]]  # Handle case with no molecules)
                tanimoto_distance = pd.concat(
                    [
                        tanimoto_distance,
                        pd.DataFrame(
                            td, columns=self.comparison_columns + ["TanimotoDistance"]
                        ),
                    ]
                )
                continue
            df["Molecule"] = self._calculate_fingerprints(tuple(smiles))

            fps = df["Molecule"].tolist()

            td = self._calculate_tanimoto_distance(fps)
            td = np.nanmean(td)
            td = [group_cols + [td]]
            # td = [group_cols + [t] for t in td]
            tanimoto_distance = pd.concat(
                [
                    tanimoto_distance,
                    pd.DataFrame(
                        td, columns=self.comparison_columns + ["TanimotoDistance"]
                    ),
                ]
            )
        tanimoto_distance = tanimoto_distance.reset_index(drop=True)
        self.tanimotodistance_internal = tanimoto_distance
        self.tanimotodistance_internal = self._average_across_comparison_columns(
            self.tanimotodistance_internal
        )
        return tanimoto_distance

    def set_known_molecules(
        self,
        known_molecules: List[str],
    ) -> None:
        """
        Set reference SMILES for known molecules.

        Stores known molecules for later novelty comparison. Used by
        get_novelty_tanimoto_distance to measure how different generated
        molecules are from known reference compounds.

        Args:
            known_molecules (List[str]): List of SMILES strings representing known/reference molecules

        Returns:
            None
        """
        self.known_molecules = known_molecules

    def get_novelty_tanimoto_distance(
        self, known_molecules: List[str]
    ) -> pd.DataFrame:
        """
        Calculate Tanimoto distance between hit molecules and known reference set.

        Measures novelty by computing the minimum Tanimoto distance from each hit
        molecule to the known molecule set. Smaller distance = less novel.

        Args:
            known_molecules (List[str]): List of SMILES strings for reference molecules

        Returns:
            pd.DataFrame: Minimum distance to known molecules per hit, columns:
                - comparison_columns: Grouping columns
                - KnownMoleculesTanimotoDistance: Minimum Tanimoto distance to known set

        Note:
            Returns empty DataFrame if no hit molecules exist.
            Final result is averaged across comparison_columns.
        """
        moo_dataframe = self.moo_dataframe[self.hit_molecules]
        if len(moo_dataframe) == 0:
            self.tanimotodistance_novelty = pd.DataFrame(
                columns=self.comparison_columns + ["KnownMoleculesTanimotoDistance"]
            )
            return self.tanimotodistance_novelty
        tanimoto_distance = pd.DataFrame()
        known_molecules_fp = self._calculate_fingerprints(tuple(known_molecules))

        for group_cols, df in moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )
            df["Molecule"] = self._calculate_fingerprints(
                tuple(df[self.smiles_col].tolist())
            )

            fps = df["Molecule"].tolist()
            td = np.array(
                [
                    1
                    - np.array(
                        cDataStructs.BulkTanimotoSimilarity(fps[i], known_molecules_fp)
                    )
                    for i in range(len(fps))
                ]
            )
            tan_dist = [
                group_cols + [np.nanmin(t)] for t in td
            ]  # TODO: Check this is the correct data dtructure
            tanimoto_distance = pd.concat(
                [
                    tanimoto_distance,
                    pd.DataFrame(
                        tan_dist,
                        columns=self.comparison_columns
                        + ["KnownMoleculesTanimotoDistance"],
                    ),
                ]
            )

        tanimoto_distance = tanimoto_distance.reset_index()
        tanimoto_distance = self._average_across_comparison_columns(tanimoto_distance)
        self.tanimotodistance_novelty = tanimoto_distance

        return tanimoto_distance

    def get_pareto_molecules(self) -> pd.DataFrame:
        """
        Identify and count Pareto optimal molecules from hit set.

        Uses non-dominated sorting to find molecules on the Pareto front for each
        MOO method (within each comparison group). Returns proportions of molecules
        per MOO method that are on the front.

        Returns:
            pd.DataFrame: Proportion of Pareto optimal molecules, columns:
                - comparison_columns (except moo_col): Grouping columns
                - moo_col: MOO method identifier
                - count: Proportion of molecules on Pareto front (0-1)

        Note:
            Handles edge case of ≤1 molecule per group gracefully.
        """
        pareto_molecules = pd.DataFrame()
        moo_dataframe = self.moo_dataframe[self.hit_molecules]
        comparison_columns_across = self.comparison_columns.copy()
        comparison_columns_across.remove(
            self.moo_col
        ) if self.moo_col in comparison_columns_across else None
        for group_cols, df in moo_dataframe.groupby(comparison_columns_across):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )

            if len(df) <= 1:
                df["NDS"] = 0
                df = pd.DataFrame(
                    df[df["NDS"] == 0].value_counts(self.moo_col, normalize=True)
                ).reset_index()
                df[comparison_columns_across] = group_cols
                pareto_molecules = pd.concat([pareto_molecules, df])
                continue

            ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(
                -1 * df[self.objectives].values
            )
            df["NDS"] = ndl
            df = pd.DataFrame(
                df[df["NDS"] == 0].value_counts(self.moo_col, normalize=True)
            ).reset_index()
            df[comparison_columns_across] = group_cols
            pareto_molecules = pd.concat([pareto_molecules, df])

        self.pareto_molecules = pareto_molecules
        return pareto_molecules

    def get_hypervolume(self) -> pd.DataFrame:
        """
        Calculate hypervolume indicator for Pareto front of each MOO method.

        Hypervolume measures the volume of objective space dominated by the Pareto set,
        providing a scalar quality measure for multi-objective optimization. Uses
        Pareto front from hit molecules. Reference point is origin (all zeros).

        Returns:
            pd.DataFrame: Hypervolume values per comparison group, columns:
                - comparison_columns: Grouping columns
                - Hypervolume: Hypervolume value (0 for ≤1 molecule)

        Note:
            Uses hit_molecules mask; run define_hitmolecules first.
        """
        hypervolume_bestmols = []
        moo_dataframe = self.moo_dataframe[self.hit_molecules]
        reference_point = np.zeros(len(self.objectives))
        for group_cols, df in moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )
            if len(df) <= 1:
                hypervolume_bestmols.append(group_cols + [0])
                continue
            ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(
                -1 * df[self.objectives].values
            )
            df["NDS"] = ndl
            hv = hypervolume(-df.loc[df["NDS"] == 0, self.objectives].values)
            hv = hv.compute(reference_point, hv_algo=pg.bf_fpras())
            hypervolume_bestmols.append(group_cols + [hv])

        self.hypervolume_bestmols = pd.DataFrame(
            hypervolume_bestmols, columns=self.comparison_columns + ["Hypervolume"]
        )
        return self.hypervolume_bestmols

    def calculate_hitrate_thresholds(self) -> None:
        """
        Calculate minimum objective value threshold for each relevant molecule.

        For each molecule, finds the minimum value across all objectives. This value
        represents the "bottleneck" objective limiting hit rate determination.

        Sets a 'Threshold' column equal to the minimum objective value for relevant
        molecules, and 0 for non-relevant molecules.

        Returns:
            None

        Note:
            Requires define_relevant_molecules to be called first.
        """
        self.moo_dataframe["Threshold"] = 0
        self.moo_dataframe.loc[self.relevant_molecules, "Threshold"] = np.min(
            self.moo_dataframe.loc[self.relevant_molecules, self.objectives].values,
            axis=1,
        )

    def get_unique_moo_molecules(self) -> pd.DataFrame:
        """
        Identify molecules unique to specific MOO methods.

        Finds hit molecules that are discovered by only one MOO method (not counting
        replicates). Calculates proportions of unique vs. shared molecules per method.

        Returns:
            pd.DataFrame: Unique molecule proportions, columns:
                - comparison_columns: Grouping columns
                - Unique: Boolean, whether molecule is unique to one method
                - proportion: Float, proportion in category
                - HitRateUniqPercent: Percentage (0-100)

        Note:
            Excludes replicate_col from uniqueness definition (molecules can be
            unique across replicates but shared across methods).
        """
        cc = self.comparison_columns.copy()

        if self.replicate_col is not None:
            cc.remove(self.replicate_col)

        df_uniq = self.moo_dataframe.loc[self.hit_molecules].copy()
        df_uniq["Unique"] = ~self.moo_dataframe.duplicated(
            [self.smiles_col] + cc, keep=False
        )

        df_uniq = pd.DataFrame(
            df_uniq.groupby(self.comparison_columns).value_counts(
                ["Unique"], normalize=True
            )
        ).reset_index()
        all_combinations = pd.MultiIndex.from_product(
            [
                self.moo_dataframe[c].unique()
                if c in self.moo_dataframe
                else [True, False]
                for c in self.comparison_columns + ["Unique"]
            ],
            names=self.comparison_columns + ["Unique"],
        )

        df_uniq = df_uniq.set_index(self.comparison_columns + ["Unique"])
        df_uniq = df_uniq.reindex(all_combinations, fill_value=0).reset_index()
        df_uniq["HitRateUniqPercent"] = df_uniq["proportion"] * 100
        return df_uniq

    def calculate_pareto_front(
        self, M: np.ndarray, len_x: int
    ) -> Tuple[List[int], int]:
        """
        Calculate first Pareto front using domination matrix.

        Implements non-dominated sorting using a domination matrix representation.
        Identifies all solutions that are not dominated by any other solution.

        Args:
            M (np.ndarray): Domination matrix where M[i,j] > 0 means i dominates j,
                M[i,j] < 0 means j dominates i, M[i,j] = 0 means neither dominates
            len_x (int): Number of solutions to consider (length of problem)

        Returns:
            Tuple[List[int], int]: 
                - current_front: Indices of non-dominated solutions
                - n_ranked: Number of solutions in front
        """
        n = M.shape[0]

        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        M[np.tril_indices(n)] = 0
        for i in range(len_x):
            idx = np.argwhere(M[i] >= 1)
            if len(idx) > 0:
                j = list(np.hstack(idx))
                is_dominating[i].extend(j)
                n_dominated[j] += 1

            j = list(np.argwhere(M[i] <= -1))
            if len(j) > 0:
                j = np.hstack(j)
                for j_ in j:
                    is_dominating[j_].append(i)
                n_dominated[i] += 1
            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1
        return current_front, n_ranked

    def set_known_molecule_properties(
        self,
        known_molecule_properties: Dict,
    ) -> None:
        """
        Set reference objective values for known molecules.

        Stores known molecule properties for comparison with generated molecules.
        Used by get_nondominated_solutions to identify generated molecules not
        dominated by reference compounds.

        Args:
            known_molecule_properties (Dict): Dictionary of objective values for known molecules.
                Keys should be tuples matching comparison_columns order.
                Values should be arrays of objective values.
                Example: {('scaffold_A', 'moo_1'): [0.8, 0.9, 0.7]}

        Returns:
            None
        """
        self.known_molecule_properties = known_molecule_properties

    def get_nondominated_solutions(
        self, known_molecule_properties: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Find non-dominated solutions compared to known molecules.

        Uses non-dominated sorting to identify generated molecules that are not dominated
        by any known reference molecules. Compares generated hit molecules against
        a reference set for each comparison group.

        Args:
            known_molecule_properties (Optional[Dict]): Dictionary with known molecule objective values.
                If None, uses stored self.known_molecule_properties.
                Keys are tuples matching comparison_columns; values are objective arrays.

        Returns:
            pd.DataFrame: Count of non-dominated molecules per group, columns:
                - comparison_columns: Grouping columns
                - Nondominated Solutions: Count of non-dominated molecules

        Raises:
            AttributeError: If known_molecule_properties not provided and not previously set

        Logs Warnings:
            - No known molecule properties found for a group
            - Multiple matching properties found (uses first match)

        Note:
            Final result is averaged across comparison_columns.
        """
        self.nondominated_solutions = pd.DataFrame()

        if known_molecule_properties is None:
            known_molecule_properties = self.known_molecule_properties

        known_molecule_properties_keys = list(known_molecule_properties.keys())
        for group_cols, df in self.moo_dataframe.groupby(self.comparison_columns):
            group_cols = (
                list(group_cols) if isinstance(group_cols, tuple) else [group_cols]
            )
            df = df.drop_duplicates(self.smiles_col)
            df = df[df["Filters"]]
            df.reset_index(drop=True, inplace=True)
            X = df[self.objectives].values

            known_molecule_properties_select = [
                any(a == b for a, b in zip(group_cols, k) if b is not None)
                for k in known_molecule_properties_keys
            ]

            if known_molecule_properties_select.count(True) == 0:
                log.warning(
                    f"No known molecule properties found for group {group_cols}, skipping non-dominated calculation"
                )
                # if as_generator:
                #     yield group_cols, pd.DataFrame(columns=df.columns), 0
                continue
            elif known_molecule_properties_select.count(True) > 1:
                log.warning(
                    f"Multiple known molecule properties found for group {group_cols}, using first match"
                )
            known_molecule_properties_select = known_molecule_properties_select.index(
                True
            )

            X_jnt = np.vstack(
                [
                    X,
                    known_molecule_properties[
                        known_molecule_properties_keys[known_molecule_properties_select]
                    ],
                ]
            )
            ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(-1 * X_jnt)

            cf = [True if (d > len(X)).sum() == 0 else False for d in dl[: len(X)]]
            n_ranked = sum(cf)

            self.nondominated_solutions = pd.concat(
                [
                    self.nondominated_solutions,
                    pd.DataFrame(
                        [group_cols + [n_ranked]],
                        columns=self.comparison_columns + ["Nondominated Solutions"],
                    ),
                ]
            )

        self.nondominated_solutions = self._average_across_comparison_columns(
            self.nondominated_solutions
        )

        return self.nondominated_solutions
