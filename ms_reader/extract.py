"""Module containing the parser for tracefinder data and handling all the
logic of MS_Reader """

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from natsort import natsorted, index_natsorted
from openpyxl import load_workbook

from ms_reader.utils import format


class Extractor:

    def __init__(self, data, calrep=None, metadata=None, met_class="CM"):

        self.normalised_ratios = None
        self.norm_c12_areas = None
        self.quantities = None
        self.normalised_quantities = None
        self.qc_table = None
        self.excluded_c13_areas = None
        self.excluded_c12_areas = None
        self.c13_areas = None
        self.c12_areas = None
        self.ratios = None
        self.no_ratio = None
        self.cal_nulls = None
        self.calib_data = None
        self.calib_nulls = None
        self.loq_table = None
        self.missing_cal_points = None
        self.concentration_table = None
        self._concunits = None

        self.stream = io.StringIO()
        handle = logging.StreamHandler(self.stream)
        handle.setLevel(logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handle)
        self.logger.setLevel(logging.INFO)

        # List to keep the different generated tables
        self.excel_tables = []

        self.data = Extractor._read_data(data)
        if calrep is None:
            self.calrep = None
        else:
            self.calrep = Extractor._read_data(calrep)
        if metadata is None:
            self.metadata = None
            self.unit = "µM"  # Default unit, no metadata is present so no
            # normalisation (expressed in concentration)
        else:
            self.metadata = Extractor._read_data(metadata)
            self.unit = "µmol"  # Default unit, if metadata is present data
            # is normalised (expressed in quantities)

        self.data["Sample_Name"] = self.data["Filename"]

        # If metadata file is given, check that the sample names are the
        # same as in the data
        if self.metadata is not None:
            if natsorted(list(self.metadata.Sample_Name)) \
                    != natsorted(list(self.data.Sample_Name.unique())):
                raise ValueError("The Sample names in the data and metadata "
                                 "do not correspond. Please check them and "
                                 "try again.")

            self.metadata.set_index("Sample_Name", inplace=True)
            self._check_metadata_columns()
            self.excel_tables.append(("Metadata", self.metadata))

        self.data.drop("Filename", axis=1, inplace=True)
        columns = [
            "Compound", "Sample_Name", "Area", "Sample Type",
            "Calculated Amt", "Theoretical Amt",
            "Response Ratio", "Excluded", "%Diff"
        ]
        self.data = self.data[columns]
        self._replace_nf()
        self._split_dataframes_test()
        self._get_excluded()

        self.met_class = met_class

    @property
    def norm_unit(self):

        if self._concunits is not None:
            return self._concunits
        elif self.md_units is not None:
            _concunits = [x for x in self.md_units.iloc[0, 1:]]
            self._concunits = "/".join(_concunits)
            return self._concunits
        else:
            raise AttributeError(
                f"The metadata has not been loaded in properly. Cannot parse "
                f"units from file. "
                f"Metadata={self.metadata}"
            )

    def _check_metadata_columns(self):
        """
        Perform checks on the headers in the metadata file
        :return:
        """

        # Check that volume normalisation columns are present
        for col in ["Resuspension_Volume", "Volume_Unit"]:
            if col not in self.metadata.columns:
                raise ValueError(f"{col} is missing from the metadata file "
                                 f"columns")
        md_values_cols = ["Resuspension_Volume"]
        md_unit_cols = ["Volume_Unit"]
        # Check if there are other normalisations to make,
        # and if so if they are well paired
        if len(self.metadata.columns) > 2:
            col_nums = []
            number_cols = int((len(self.metadata.columns) - 2) / 2)
            for x in range(1, number_cols + 1):
                col_nums += 2 * [x]
            for idx, (num, col) in enumerate(
                    zip(col_nums, self.metadata.columns[2:])
            ):
                if idx % 2 == 0:
                    if col != f"Norm{num}":
                        raise ValueError(
                            f'The column "{col}" is not right format. '
                            f'Expected format: "Norm{num}"'
                        )
                    md_values_cols.append(col)
                else:
                    if col != f"Norm{num}_Unit":
                        raise ValueError(
                            f'The column "{col}" is not right format. '
                            f'Expected format: "Norm{num}_Unit"'
                        )
                    md_unit_cols.append(col)

        self.md_units = self.metadata[md_unit_cols]
        self.md_values = self.metadata[md_values_cols]

        # Convert all md_values to numeric and intercept any conversion errors
        # which might mean some strings are present
        try:
            self.md_values.apply(
                lambda s: pd.to_numeric(s, errors="raise")
            )
        except ValueError:
            raise TypeError(
                "Error while converting to numeric values. "
                "Are you sure your metadata file contains only numbers?"
            )
        except Exception:
            raise RuntimeError(
                "Unknown error while converting to numeric values."
            )
        else:
            # Handle NaNs
            self.md_values.fillna(1, inplace=True)

        # Make sure units are the same for all samples
        for col in self.md_units.columns:
            if len(self.md_units[col].unique()) != 1:
                raise ValueError(f"The column {col} has more than one unit")

    def generate_metadata(self, nb_norms: int = 1) -> pd.DataFrame:
        """
        Generate metadata file from input data.

        :return: Dataframe containing metadata
        """

        if nb_norms < 0 or isinstance(nb_norms, float):
            raise ValueError("Normalisation number must be a positive integer")
        cols = ["Resuspension_Volume", "Volume_Unit"]
        for i in range(1, nb_norms + 1):
            cols.append(f"Norm{i}")
            cols.append(f"Norm{i}_Unit")
        metadata = pd.DataFrame(columns=cols)
        metadata["Sample_Name"] = natsorted(self.data["Sample_Name"].unique())
        metadata.set_index("Sample_Name", inplace=True)
        metadata["Volume_Unit"] = "µL"
        return metadata

    # noinspection PyArgumentList
    @staticmethod
    def _read_data(path: str) -> pd.DataFrame:
        """
        Read input data
        :param path: path to the data
        :return: Pandas dataframe containing data
        """
        if type(path) is str:
            datapath = Path(path)
            if datapath.suffix == ".csv":
                data = pd.read_csv(datapath, sep=";")
            elif datapath.suffix == ".xlsx":
                data = pd.read_excel(datapath)
            else:
                raise ValueError(
                    "Data file format not recognized. "
                    "Accepted formats: csv, xlsx"
                )
        else:
            data = path
        return data

    @staticmethod
    def get_min(row: pd.Series) -> np.ndarray:
        """
        Get row minimum (used for applying along axis in a pandas dataframe
        using the apply method)
        :param row: input row passed in by apply method
        :return minimum_value: Row minimum value (ndarray scalar)
        """
        row = row.values
        only_num = [val for val in row if type(val) != str]
        if np.isnan(only_num).all():
            return np.nan
        else:
            minimum_val = np.nanmin(only_num)
            return minimum_val

    @staticmethod
    def get_max(row: pd.Series) -> np.ndarray:
        """
        Get row maximum (used for applying along axis in a pandas dataframe
        using the apply method)
        :param row: input row passed in by apply method
        :return minimum_value: Row maximum value (ndarray scalar)
        """
        row = row.values
        only_num = [val for val in row if type(val) != str]
        if np.isnan(only_num).all():
            return np.nan
        else:
            maximum_val = np.nanmax(only_num)
            return maximum_val

    def _replace_nf(self):
        """
        Replace initial N/F with 0
        :return: None
        """
        self.data.loc[:, "Area"] = \
            self.data.loc[:, "Area"].replace("N/F", 0).copy()
        self.data.loc[:, "Calculated Amt"] = \
            self.data.loc[:, "Calculated Amt"].replace("N/F", 0).copy()

    def _split_dataframes(self):
        """
        Split the original dataframe into sub-dataframes of each content type
        :return: None
        """

        self.calib_data = self.data[
            self.data["Sample Type"].str.contains("Cal")
        ]
        self.sample_data = self.data[
            self.data["Sample Type"].str.contains("Unknown")
        ]
        self.qc_data = self.data[
            self.data["Sample Type"].str.contains("QC")
        ]
        self.anal_blank = self.data[
            self.data["Sample Type"].str.contains("Solvent")
        ]
        self.data_blank = self.data[
            self.data["Sample Type"].str.contains("Matrix Blank")
        ]

    def _split_dataframes_test(self):

        self.calib_data = self.data[
            self.data["Sample Type"].str.contains("Cal")
        ].copy()
        self.sample_data = self.data[
            ~self.data["Sample Type"].str.contains("Cal|QC")
        ].copy()
        self.qc_data = self.data[
            self.data["Sample Type"].str.contains("QC")
        ].copy()
        self.blank_data = self.data[
            self.data["Sample_Name"].str.contains("Blank")
        ].copy()
        self.sample_data = self.sample_data[
            ~self.sample_data["Sample_Name"].str.contains("Blank")
        ].copy()

    def _get_excluded(self):
        """
        Replace excluded points concentration with 'excluded'
        :return: None
        """
        self.calib_data.loc[:, "Calculated Amt"] = pd.to_numeric(
            self.calib_data["Calculated Amt"]
        )
        self.calib_data.loc[
            self.calib_data["Excluded"] == "True", "Calculated Amt"
        ] = "Excluded"

    def _generate_minmax_calib(self):
        """
        Generate calibration dataframe with minimum and maximum value columns
        :return: None
        """

        # Isolate the C12 data from the internal standard (C13)
        min_max_calib = self.calib_data[
            ~self.calib_data["Compound"].str.contains("C13")
        ]
        min_max_calib = min_max_calib.sort_values(
            ["Compound", "Sample_Name"],
            key=lambda x: np.argsort(
                index_natsorted(zip(
                    min_max_calib["Compound"],
                    min_max_calib["Sample_Name"]
                ))
            )
        )
        # Pivot and add the min and max columns
        min_max_calib = min_max_calib.pivot(
            index="Compound",
            columns="Sample_Name",
            values="Calculated Amt"
        )
        min_max_calib = min_max_calib.reindex(
            columns=natsorted(min_max_calib.columns)
        )
        calibration_minimum = min_max_calib.apply(Extractor.get_min, axis=1)
        calibration_maximum = min_max_calib.apply(Extractor.get_max, axis=1)
        min_max_calib["min"] = calibration_minimum
        min_max_calib["max"] = calibration_maximum
        self.min_max = min_max_calib[["min", "max"]].copy()
        self.min_max = self.min_max.rename(
            {"min": "LLOQ", "max": "ULOQ"},
            axis=1
        )
        self.calib_data = min_max_calib
        self.excel_tables.append(
            ("Calibration", self.calib_data)
        )

    def handle_calibration(self):
        """
        Call the generate minmax calib function to handle the calibration df
        :return: None
        """
        self._generate_minmax_calib()
        self.calib_data, self.calib_nulls = self._replace(
            self.calib_data,
            to_replace=[np.nan, 0],
            value="NA",
            axis="row",
            drop=True
        )
        if not self.calib_nulls.empty:
            self.logger.info(
                f"\nRows removed from the calibration table:"
                f"\n{self.calib_nulls.T}"
            )

    def handle_qc(self) -> bool:
        """
        Get run quality control. If False the run is invalidated.
        :return: Quality control (bool)
        """

        if self.met_class == "CM":
            qc_mets = ["FruBP", "Orotate", "Rib1P"]
            qc_verif = self.qc_data[
                self.qc_data["Compound"].isin(qc_mets)
            ].copy()
        elif self.met_class == "AA" or self.met_class == "CoA":
            qc_verif = self.qc_data[
                ~self.qc_data["Compound"].str.contains("C13")
            ].copy()
        else:
            raise KeyError(
                "The selected metabolite class is not valid. "
                "Valid classes are CM for Central Metabolites, "
                "AA for Amino Acids and CoA for Coenzymes A"
            )

        # Replace commas with dots and convert columns to numeric
        for col in ["%Diff", "Theoretical Amt", "Calculated Amt"]:
            qc_verif[col] = qc_verif[col].apply(
                lambda x: str(x).replace(",", ".")
            )
            qc_verif[col] = pd.to_numeric(qc_verif[col], errors="coerce")

        self.qc_table = qc_verif[
            ["Compound", "Theoretical Amt", "Calculated Amt", "%Diff"]
        ]
        self.qc_table.set_index("Compound", inplace=True)

        # QC is set to have a difference between sample point
        # and QC of 20% maximum
        if (abs(qc_verif["%Diff"].values) > 20).any():
            qc = False
        else:
            qc = True
        self.qc_table = self.qc_table.astype(str)

        self.qc_table = self.qc_table.style.apply(self._color_qc, axis=1,
                                                  subset=["%Diff"])

        self.excel_tables.append(("Quality Control", self.qc_table))
        return qc

    def generate_areas_table(self):
        """
        Generate the 12C and 13C area tables by separating them from the
        rest of the data and formatting
        :return: None
        """

        # separate 12C and 13C sample data
        c12 = self.sample_data[
            ~self.sample_data["Compound"].str.contains("C13")
        ].copy()
        c13 = self.sample_data[
            self.sample_data["Compound"].str.contains("C13")
        ].copy()

        c12_areas = pd.pivot_table(c12, "Area", "Compound", "Sample_Name")
        c13_areas = pd.pivot_table(c13, "Area", "Compound", "Sample_Name")

        # get rid of rows where all data points are excluded
        self.c12_areas = c12_areas[~c12_areas.isin(["Excluded"]).all(axis=1)]
        self.c13_areas = c13_areas[~c13_areas.isin(["Excluded"]).all(axis=1)]

        self.excluded_c12_areas = c12_areas[
            c12_areas.isin(["Excluded"]).all(axis=1)
        ]
        self.excluded_c13_areas = c13_areas[
            c13_areas.isin(["Excluded"]).all(axis=1)
        ]

        # rearrange and normalise
        c12_cols = natsorted(self.c12_areas.columns)
        c13_cols = natsorted(self.c13_areas.columns)
        c12_cols.insert(0, "Unit")
        c13_cols.insert(0, "Unit")
        base_unit = "area"

        # Normalise c12 data
        if self.metadata is not None:
            self.norm_c12_areas = self.normalise(self.c12_areas.copy())
            self.norm_c12_areas = self.norm_c12_areas.applymap(format)
            self.norm_c12_areas["Unit"] = f"{base_unit}/{self.norm_unit}"
            self.norm_c12_areas = self.norm_c12_areas[c12_cols]

        self.c12_areas = self.c12_areas.applymap(format)
        self.c13_areas = self.c13_areas.applymap(format)
        self.c12_areas["Unit"] = base_unit
        self.c13_areas["Unit"] = base_unit
        self.c12_areas = self.c12_areas[c12_cols]
        self.c13_areas = self.c13_areas[c13_cols]

        self.excel_tables.append(
            ("C12_areas", self.c12_areas)
        )
        self.excel_tables.append(
            ("C13_areas", self.c13_areas)
        )
        if self.metadata is not None:
            self.excel_tables.append(
                ("Normalised_C12_areas", self.norm_c12_areas)
            )

    def normalise(self, df, multiply=True, divide=True) -> pd.DataFrame:
        """
        Normalise concentrations by multiplying by volume and dividing by norms

        :param df: dataframe to normalise
        :param multiply: should the multiplications be applied
        :param divide: should the divisions be applied
        :return: dataframe containing normalised data
        """

        df = df.apply(
            lambda s: pd.to_numeric(s, errors="raise")
        )

        for col in df.columns:
            if multiply:
                df[col] = df[col].multiply(
                    self.md_values.at[col, "Resuspension_Volume"])
                df[col] = df[col].multiply(1e-6)
            if divide:
                for norm in self.md_values.columns[1:]:
                    df[col] = df[col].divide(self.md_values.at[col, norm])
        return df

    def _generate_normalised_concentrations(self, columns: list):
        """
        Normalise the concentrations and put them into a clean df. Also
        generate loq_table

        :param columns: columns to use for sorting the tables
        :return: None
        """

        self.quantities = self.normalise(
            self.concentration_table.copy(),
            divide=False
        )
        self.normalised_quantities = self.normalise(
            self.concentration_table.copy()
        )
        # sort the columns naturally
        self.quantities = self.quantities[columns]
        self.normalised_quantities = self.normalised_quantities[columns]
        # define loqs
        self.loq_table = self.normalised_quantities.copy()
        for idx in self.loq_table.index:
            lloq_mask = self.concentration_table.loc[idx, :].apply(
                lambda x: float(x) < self.calib_data.at[idx, "min"]
            )
            uloq_mask = self.concentration_table.loc[idx, :].apply(
                lambda x: float(x) > self.calib_data.at[idx, "max"]
            )
            self.loq_table.loc[idx, :] = self.loq_table.loc[idx, :].where(
                ~lloq_mask, other="<LLOQ"
            )
            self.loq_table.loc[idx, :] = self.loq_table.loc[idx, :].where(
                ~uloq_mask, other=">ULOQ"
            )

    def _handle_conc_norm(self, cols, base_unit):
        """
        Handle normalisations on the concentrations tables

        :param cols: columns to rearrange the dfs
        :param base_unit: unit to put in unit column
        :return: None
        """
        # normalise
        self._generate_normalised_concentrations(cols)
        # map "ND" to negative and null values
        self.quantities = self.quantities.applymap(format)
        self.normalised_quantities = \
            self.normalised_quantities.applymap(format)
        self.loq_table = self.loq_table.mask(
            self.normalised_quantities == "ND", "ND"
        )
        # add unit column
        self.quantities["Unit"] = base_unit
        self.normalised_quantities["Unit"] = \
            f"{base_unit}/{self.norm_unit}"
        self.loq_table["Unit"] = f"{base_unit}/{self.norm_unit}"
        cols.insert(0, "Unit")
        self.quantities = self.quantities[cols]
        self.normalised_quantities = self.normalised_quantities[cols]
        self.quantities = pd.merge(
            left=self.quantities,
            right=self.min_max,
            how="left",
            left_index=True,
            right_index=True
        )
        self.normalised_quantities = pd.merge(
            left=self.normalised_quantities,
            right=self.min_max,
            how="left",
            left_index=True,
            right_index=True
        )
        self.loq_table = self.loq_table[cols]

    def _handle_conc_no_norm(self, cols, base_unit):
        """
        Handle the concentrations tables

        :param cols: columns to rearrange the dfs
        :param base_unit: unit to put in unit column
        :return: None
        """

        # generate loq_table
        self.loq_table = self._define_loq(self.concentration_table.copy())
        # map "ND" to negative and null values
        self.concentration_table = self.concentration_table.applymap(
            format
        )
        self.loq_table = self.loq_table.mask(
            self.concentration_table == "ND", "ND"
        )
        # add unit column
        cols.insert(0, "Unit")
        self.concentration_table["Unit"] = base_unit
        self.loq_table["Unit"] = base_unit
        self.concentration_table, self.loq_table = \
            self.concentration_table[cols], self.loq_table[cols]
        self.concentration_table = pd.merge(
            left=self.concentration_table,
            right=self.min_max,
            how="left",
            left_index=True,
            right_index=True
        )

    def generate_concentrations_table(self, loq_export, base_unit=None):

        if base_unit is None:
            base_unit = "µmol" if self.metadata is not None else "µM"

        # Isolate the C12 data
        concentrations = self.sample_data[
            ~self.sample_data["Compound"].str.contains("C13")
        ]
        # transpose the data
        self.concentration_table = concentrations.pivot(
            index="Compound",
            columns="Sample_Name",
            values="Calculated Amt"
        )
        # Replace nans and nulls with "NA"
        if not self.calib_nulls.empty:
            self.concentration_table.drop(
                index=list(self.calib_nulls.columns),
                inplace=True
            )
        # sort the columns naturally
        new_cols = natsorted(self.concentration_table.columns)
        self.concentration_table = self.concentration_table[new_cols]
        # If metadata detected, normalise the concentrations and do loq,
        # else only do loq
        if self.metadata is not None:
            self._handle_conc_norm(new_cols, base_unit)
        else:
            self._handle_conc_no_norm(new_cols, base_unit)
        self.loq_table = pd.merge(
            left=self.loq_table,
            right=self.min_max,
            how="left",
            left_index=True,
            right_index=True
        )
        # Add to export lists
        if self.metadata is not None:
            self.excel_tables.append(
                ("Quantities", self.quantities)
            )
            self.excel_tables.append(
                ("Normalised_Quantities", self.normalised_quantities),
            )
            if loq_export:
                if self.metadata is not None:
                    self.excel_tables.append(
                        ("Normalised_Quantities_LLOQ", self.loq_table)
                    )
        else:
            self.excel_tables.append(
                ("Concentrations", self.concentration_table),
            )
            if loq_export:
                self.excel_tables.append(
                    ("Concentrations_LLOQ", self.loq_table)
                )

    def generate_ratios(self):

        # Isolate missing c13 compounds
        c12 = self.sample_data[
            ~self.sample_data["Compound"].str.contains("C13")].copy()
        c13 = self.sample_data[
            self.sample_data["Compound"].str.contains("C13")].copy()
        c12_compounds, missing_c13_std = self._check_if_std(
            list(c12["Compound"].unique()), list(c13["Compound"].unique())
        )
        if missing_c13_std:
            self.logger.info(
                f"Metabolites missing from IDMS: \n{missing_c13_std}")
        else:
            self.logger.info("All metabolites are present in the IDMS")

        # Drop missing compounds from c12 df
        c13.loc[:, "Compound"] = c13.loc[:, "Compound"].str.slice(0, -4)
        c12.set_index(["Compound", "Sample_Name"], inplace=True)
        c13.set_index(["Compound", "Sample_Name"], inplace=True)
        c12.sort_index(level=['Compound', 'Sample_Name'], inplace=True)
        c13.sort_index(level=['Compound', 'Sample_Name'], inplace=True)
        if missing_c13_std:
            c12.drop(missing_c13_std, inplace=True)
        to_log = pd.pivot_table(
            c13[c13['Area'] == 0], 'Area', 'Compound', 'Sample_Name'
        )
        self.logger.warning(
            f"\nMetabolites with null areas in c13 data:\n"
            f"{to_log}\n")

        # Ensure that c12 and C13 have same indexes. Check both ways and
        # isolate missing indexes. Compute ratios
        if c12.index.difference(c13.index).levshape != (
                0, 0) and c13.index.difference(c12.index).levshape != (0, 0):
            c12_diff = c12.index.difference(c13.index)
            c13_diff = c13.index.difference(c12.index)
            intercept = c12.index.intersection(c13.index)
            self.ratios = c12.loc[intercept, "Area"].divide(
                c13.loc[intercept, "Area"])
            self.ratios.name = "Ratios"
            self.no_ratio = {
                "c12": c12.loc[c12_diff, :],
                "c13": c13.loc[c13_diff, :]
            }
            self.logger.debug(
                f"Some index levels are in C12 data and not in C13 data. "
                f"Differences:\n{self.no_ratio['c12']} "
                f"\n Some index levels are in C13 data and not in C12 data. "
                f"Differences: \n{self.no_ratio['c13']}")
        else:
            if c12.index.difference(c13.index).levshape != (0, 0):
                c12_c13_diff = c12.index.difference(c13.index)
                self.ratios = c12.drop(c12_c13_diff).loc[
                    c12_c13_diff, "Area"].divide(c13.loc[:, "Area"])
                self.ratios.name = "Ratios"
                self.no_ratio = c12.loc[c12_c13_diff, :]
                self.logger.info(
                    f"Some index levels are in C12 data and not in C13 data. "
                    f"Differences:\n{c12_c13_diff}"
                )
                print(f"Ratios calculated:\n{self.ratios}")
            elif c13.index.difference(c12.index).levshape != (0, 0):
                c13_c12_diff = c13.index.difference(c12.index)
                self.ratios = c12.loc[:, "Area"].divide(
                    c13.drop(c13_c12_diff).loc[:, "Area"])
                self.ratios.name = "Ratios"
                self.no_ratio = c13.loc[c13_c12_diff, :]
                self.logger.info(
                    f"Some index levels are in C13 data and not in C12 data. "
                    f"Differences:\n{c13_c12_diff}"
                )
                print(f"Ratios calculated:\n{self.ratios}")
            else:
                self.ratios = c12.loc[:, "Area"].divide(c13.loc[:, "Area"])
                self.ratios.name = "Ratios"
                print(
                    f"Ratios calculated with no differences detected between "
                    f"c12 and c13 indexes. Ratios:\n{self.ratios}"
                )
        self.ratios = self.ratios.reset_index(level="Sample_Name")
        self.ratios = pd.pivot_table(self.ratios, "Ratios", "Compound",
                                     "Sample_Name")
        base_unit = "12C/13C"
        new_cols = natsorted(self.ratios.columns)
        new_cols.insert(0, "Unit")
        if self.metadata is not None:
            self.normalised_ratios = self.normalise(
                self.ratios.copy(),
                multiply=False
            )
            self.normalised_ratios = self.normalised_ratios.applymap(format)
            self.normalised_ratios["Unit"] = f"{base_unit}/{self.norm_unit}"
            self.normalised_ratios = self.normalised_ratios[new_cols]
            self.normalised_ratios = self._replace(
                self.normalised_ratios, [np.inf, np.nan], "NA", "dataframe"
            )
        self.ratios = self.ratios.applymap(format)
        self.ratios["Unit"] = base_unit
        self.ratios = self.ratios[new_cols]
        self.ratios = self._replace(
            self.ratios, [np.inf, np.nan], "NA", "dataframe"
        )
        self.excel_tables.append(
            ("Ratios", self.ratios)
        )
        if self.metadata is not None:
            self.excel_tables.append(
                ("Normalised_Ratios", self.normalised_ratios)
            )

    @staticmethod
    def _check_if_std(c12_compounds, c13_compounds):
        """
        Get list of missing standards in the IDMS (13C) signals. If none, do
        nothing and return 12C compounds list as is. Else, remove the missing
        metabolites from the list of 12C compounds to drop them from data
        after.

        :param c12_compounds: list of all the 12C compound names in data
        :param c13_compounds: list of all the 13C compound names in data
        :return: list of 12C compounds minus the removed compounds and list
                 of the compounds missing from IDMS
        """

        trunc_c13 = [std[:-4] for std in c13_compounds]
        missing_std = [x for x in c12_compounds if x not in trunc_c13]
        if missing_std:
            for x in missing_std:
                c12_compounds.remove(x)
            return c12_compounds, missing_std
        else:
            return c12_compounds, None

    @staticmethod
    def _isolate_nulls(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        nulls_nans = []
        for col in df.columns:
            try:
                df.loc[:, col] = pd.to_numeric(df.loc[:, col])
            except ValueError:
                continue
            else:
                if df.loc[:, col].isna().all() or (df.loc[:, col] == 0).all():
                    nulls_nans.append(df.loc[:, col].copy())
                    df.drop(col, axis=1, inplace=True)
        null_data = pd.concat(nulls_nans, axis=1)
        return df, null_data

    def _define_loq(self, loq_table):
        """
        Define a mask for <LLOQ and >ULOQ and apply the strings at the right
        places. Have option to just return the masks.
        :param loq_table: table to apply LOQs to
        :return: loq_table or masks to apply for loq table
        """
        for idx in loq_table.index:
            lloq_mask = loq_table.loc[idx, :].apply(
                lambda x: float(x) < self.calib_data.at[idx, "min"])
            uloq_mask = loq_table.loc[idx, :].apply(
                lambda x: float(x) > self.calib_data.at[idx, "max"])
            loq_table.loc[idx, :] = loq_table.loc[idx, :].where(~lloq_mask,
                                                                other="<LLOQ")
            loq_table.loc[idx, :] = loq_table.loc[idx, :].where(~uloq_mask,
                                                                other=">ULOQ")
        return loq_table

    @staticmethod
    def _color_qc(col):
        result = []
        for val in col:
            if abs(float(val)) > 20:
                result.append("background-color: #ff6666")
            else:
                result.append("background-color:  #99ff66")
        return result

    @staticmethod
    def check_r(col):
        result = []
        for val in col:
            try:
                if 0.99 <= float(val) <= 1:
                    result.append("background-color:  #99ff66")
                else:
                    result.append("background-color: #ff6666")
            except ValueError:
                result.append("background-color: #ff6666")
            return result

    def generate_report(self, mets_to_drop):

        if self.calrep is None:
            raise ValueError(
                "Cannot generate the report because no "
                "calibration report file was detected"
            )
        report = self.calrep.iloc[10:, :8]
        report = report.iloc[:-2, [0, -1]].copy()
        report.columns = ["Compound", "R²"]
        report = report[~report["Compound"].str.contains("C13")]
        report = report.sort_values("Compound")
        report.set_index("Compound", inplace=True)
        mets_to_drop = [
            item for item in mets_to_drop
            if "C13" not in item
        ]
        report = report.drop(mets_to_drop)
        self.calrep = report.copy()
        self.calrep["R²"] = self.calrep["R²"].astype(str)
        self.calrep = self.calrep.style.apply(Extractor.check_r, axis=1,
                                              subset=["R²"])

        self.excel_tables.append(
            ("Report", self.calrep)
        )

    def _output_log(self, destination):

        self.stream.seek(0)
        with open(f"{destination}\\file.log", "w") as log:
            print(self.stream.getvalue(), file=log)

    def export_stat_output(self, path):

        dest = Path(path)
        dest = dest / "output_for_graphstat.tsv"
        stat_out = self._build_stat_output()
        stat_out.to_csv(str(dest), sep="\t", index=False, encoding='utf-8-sig')
        self._output_log(path)

    def export_final_excel(self, path):

        dest_path = Path(path)
        dest_path = dest_path / "Tables.xlsx"
        try:
            with pd.ExcelWriter(str(dest_path)) as writer:
                for (name, table) in self.excel_tables:
                    table.to_excel(writer, sheet_name=name, engine="openpyxl")
        except PermissionError:
            raise PermissionError(
                "Permission denied. Please check that the "
                "Tables.xlsx file is "
                "not open so that MS_Reader can overwrite it"
            )
        # Color the sheet to delete
        sheet_names = []
        if self.metadata is None:
            sheet_names.append("Concentrations")
        else:
            sheet_names.append("Quantities")
            sheet_names.append("Normalised_Quantities")
        for sheet in sheet_names:
            try:
                self._color_sheet_tab(sheet, str(dest_path))
            except KeyError:
                pass
            except Exception:
                raise

        # Export log
        self._output_log(str(path))
        print(f"Done exporting. Path:\n {str(dest_path)}")

    @staticmethod
    def _color_sheet_tab(sheet, path, color="red"):
        """
        color the selected sheet tab and save the file

        :param sheet: Name of the sheet
        :param path: Path to excel file
        :return: None
        """

        if color == "red":
            tab_color = "FF0000"
        else:
            tab_color = None
        wb = load_workbook(filename=str(path))
        wb[sheet].sheet_properties.tabColor = tab_color
        wb.save(str(path))
        wb.close()

    @staticmethod
    def _replace(
            df: pd.DataFrame,
            to_replace: int or str or float or list,
            value: Any,
            axis: str,
            drop: bool = False
    ) -> (pd.DataFrame, pd.DataFrame):

        """

        Homemade replace function

        :param df: dataframe in which to replace values
        :param to_replace: value(s) to replace
        :param value: value to replace with
        :param axis: axis on which to apply the method (can be whole dataframe)
        :param drop: should row or column be dropped if the replacing value
                     take the whole axis (only for axis=row or column)
        :return: replaced dataframe and removed axes if drop=True

        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{df} is not a dataframe, it is of type {type(df)}")
        else:
            if df.empty:
                raise ValueError(f"{df} is empty")
            else:
                if not isinstance(to_replace, list):
                    to_replace = [to_replace]
                removed = []
                if axis == "row":
                    for idx in df.index:
                        if (df.loc[idx, :].isin(to_replace)).all():
                            if drop:
                                df.loc[idx, :] = value
                                removed.append(df.loc[idx, :])
                                df = df.drop(idx)
                            else:
                                df.loc[idx, :] = value
                if axis == "column":
                    for col in df.columns:
                        if (df.loc[:, col].isin(to_replace)).all():
                            if drop:
                                df.loc[:, col] = value
                                removed.append(df.loc[:, col])
                                df = df.drop(col, axis=1)
                            else:
                                df.loc[:, col] = value
                if axis == "dataframe":
                    df.replace(
                        to_replace=to_replace,
                        value="NA",
                        inplace=True)
        if drop:
            try:
                if axis == "row":
                    dropped = pd.concat(removed, axis=1)
                elif axis == "column":
                    dropped = pd.concat(removed)
            except ValueError:
                dropped = pd.DataFrame(columns=df.columns)
            return df, dropped
        else:
            return df

    def _build_stat_output(self):

        to_out = []
        if isinstance(self.c12_areas, pd.DataFrame) and isinstance(
                self.c13_areas, pd.DataFrame):
            c12_areas = self._replace(self.c12_areas, [0, np.inf, "", "ND"],
                                      "NA", "dataframe")
            c13_areas = self._replace(self.c13_areas, [0, np.inf, "", "ND"],
                                      "NA", "dataframe")
            c12_areas = c12_areas.reset_index()
            c13_areas = c13_areas.reset_index()
            c12_areas = c12_areas.rename({"Compound": "Features"}, axis=1)
            c13_areas = c13_areas.rename({"Compound": "Features"}, axis=1)
            c12_areas.insert(1, "type", "C12 area")
            c13_areas.insert(1, "type", "C13 area")
            to_out.append(c12_areas)
            to_out.append(c13_areas)
        if isinstance(self.concentration_table, pd.DataFrame) or isinstance(
                self.loq_table, pd.DataFrame):
            concentrations = self.loq_table.drop(
                ["LLOQ", "ULOQ"], axis=1
            ).copy()
            concentrations = self._replace(concentrations, ["<LLOQ", "ND"],
                                           "NA", "dataframe")
            concentrations = self._replace(concentrations, [">ULOQ", "ND"],
                                           "NA", "dataframe")
            concentrations = self._replace(concentrations, "", "NA",
                                           "dataframe")
            concentrations = concentrations.reset_index()
            concentrations = concentrations.rename({"Compound": "Features"},
                                                   axis=1)
            concentrations.insert(1, "type", "concentration")
            to_out.append(concentrations)
        if isinstance(self.ratios, pd.DataFrame):
            ratios = self.ratios.reset_index()
            ratios = self._replace(ratios, [np.inf, np.nan, "", "ND"], "NA",
                                   "dataframe")
            ratios = ratios.rename({"Compound": "Features"}, axis=1)
            ratios.insert(1, "type", "C12/C13 ratios")
            to_out.append(ratios)
        stat_out = pd.concat(to_out)
        stat_out["Unit"] = stat_out["Unit"].str.replace(
            pat="/",
            repl="_per_",
            regex=False
        )
        return stat_out


class Error(Exception):
    """Base class for MS Reader exceptions"""
    pass


class QCError(Error):
    """
    Exception raised for Quality Control failures

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


if __name__ == "__main__":
    test = Extractor(
        r"C:\Users\legregam\Documents\Projets\MSReader\
        data\20210506_SOKOL_filtres_MC_quant.xlsx",
        None, "AA")
    qc_result = test.handle_qc()

    # r"C:\Users\legregam\Documents\Projets\MSReader\data\Calibration
    # Report.xlsx", r"C:\Users\legregam\Documents\Projets\MSReader\data
    # \Sample_List_test.xlsx", data.handle_calibration()
    # data.generate_concentrations_table(True) data.generate_report()
    # data.get_ratios()
