from pathlib import Path
import logging
import io

import pandas as pd
import numpy as np
from natsort import natsorted, index_natsorted
from typing import Any


class Extractor:

    def __init__(self, data_path, calrep_path=None, met_class="CM"):

        self.excluded_c13_areas = None
        self.excluded_c12_areas = None
        self.c13_areas = None
        self.c12_areas = None
        self.ratios = None
        self.no_ratio = None
        self.cal_nulls = None
        self.calib_data = None
        self.conc_nulls = None
        self.loq_table = None
        self.missing_cal_points = None
        self.concentration_table = None
        self.stream = io.StringIO()
        handle = logging.StreamHandler(self.stream)
        handle.setLevel(logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handle)
        self.logger.setLevel(logging.INFO)

        self.excel_tables = []

        self.data = Extractor._read_data(data_path)
        if calrep_path is None:
            self.calrep = None
        else:
            self.calrep = Extractor._read_data(calrep_path)
        self.data["Sample_Name"] = self.data["Filename"]
        self.data.drop("Filename", axis=1, inplace=True)
        columns = ["Compound", "Sample_Name", "Area", "Sample Type", "Calculated Amt", "Theoretical Amt",
                   "Response Ratio", "Excluded", "%Diff"]
        self.data = self.data[columns]
        self._replace_nf()
        self._split_dataframes_test()
        self._get_excluded()

        self.met_class = met_class

    def generate_metadata(self, fmt: str) -> None:
        """
        Generate metadata file from input data.

        :param fmt: export format
        :return: None
        """

        metadata = pd.DataFrame(columns=["Filename", "Sample_Name"])
        metadata["Filename"] = natsorted(self.data["Filename"].unique())
        metadata["Sample_name"] = "Collaborator Sample Name"
        if fmt == "excel":
            metadata.to_excel(r"Sample_List.xlsx", index=False)
        elif fmt == "csv":
            metadata.to_csv("Sample_List.xlsx", index=False)
        else:
            raise ValueError("Requested format is not recognized")

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
                raise ValueError("Data file format not recognized. Accepted formats: csv, xlsx")
        else:
            data = path
        return data

    @staticmethod
    def get_min(row: pd.Series) -> float:
        """
        Get row minimum (used for applying along axis in a pandas dataframe using the apply method)
        :param row: input row passed in by apply method
        :return minimum_value: Row minimum value
        """
        row = row.values
        only_num = [val for val in row if type(val) != str]
        if np.isnan(only_num).all():
            return np.nan
        else:
            minimum_val = np.nanmin(only_num)
            return minimum_val

    @staticmethod
    def get_max(row: pd.Series) -> float:
        """
        Get row maximum (used for applying along axis in a pandas dataframe using the apply method)
        :param row: input row passed in by apply method
        :return minimum_value: Row maximum value
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
        self.data.loc[:, "Area"] = self.data.loc[:, "Area"].replace("N/F", 0)
        self.data.loc[:, "Calculated Amt"] = self.data.loc[:, "Calculated Amt"].replace("N/F", 0)

    def _split_dataframes(self):
        """
        Split the original dataframe into sub-dataframes of each content type
        :return: None
        """

        self.calib_data = self.data[self.data["Sample Type"].str.contains("Cal")]
        self.sample_data = self.data[self.data["Sample Type"].str.contains("Unknown")]
        self.qc_data = self.data[self.data["Sample Type"].str.contains("QC")]
        self.anal_blank = self.data[self.data["Sample Type"].str.contains("Solvent")]
        self.data_blank = self.data[self.data["Sample Type"].str.contains("Matrix Blank")]

    def _split_dataframes_test(self):

        self.calib_data = self.data[self.data["Sample Type"].str.contains("Cal")].copy()
        self.sample_data = self.data[~self.data["Sample Type"].str.contains("Cal|QC")].copy()
        self.qc_data = self.data[self.data["Sample Type"].str.contains("QC")].copy()
        self.blank_data = self.data[self.data["Sample_Name"].str.contains("Blank")].copy()
        self.sample_data = self.sample_data[~self.sample_data["Sample_Name"].str.contains("Blank")].copy()

    def _get_excluded(self):
        """
        Replace excluded points concentration with 'excluded'
        :return: None
        """
        self.calib_data.loc[:, "Calculated Amt"] = pd.to_numeric(self.calib_data["Calculated Amt"])
        self.calib_data.loc[self.calib_data["Excluded"] == True, "Calculated Amt"] = "Excluded"

    def _generate_minmax_calib(self):
        """
        Generate calibration dataframe with minimum and maximum value columns
        :return: None
        """

        min_max_calib = self.calib_data[~self.calib_data["Compound"].str.contains("C13")]
        min_max_calib = min_max_calib.sort_values(["Compound", "Sample_Name"],
                                                  key=lambda x: np.argsort(
                                                      index_natsorted(zip(min_max_calib["Compound"],
                                                                          min_max_calib["Sample_Name"]))
                                                  ))
        min_max_calib = min_max_calib.pivot(index="Compound", columns="Sample_Name", values="Calculated Amt")
        min_max_calib = min_max_calib.reindex(columns=natsorted(min_max_calib.columns))
        min_max_calib["min"] = min_max_calib.apply(Extractor.get_min, axis=1)
        min_max_calib["max"] = min_max_calib.apply(Extractor.get_max, axis=1)
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

    def handle_qc(self):
        """
        Get run quality control. If False the run is invalidated.
        :return: Quality control (bool)
        """

        if self.met_class == "CM":
            qc_mets = ["FruBP", "Oro", "Rib1P"]
            qc_verif = self.qc_data[self.qc_data["Compound"].isin(qc_mets)].copy()
        elif self.met_class == "AA" or self.met_class == "CoA":
            qc_verif = self.qc_data[~self.qc_data["Compound"].str.contains("C13")].copy()
        else:
            raise KeyError("The selected metabolite class is not valid. Valid classes are CM for Central Metabolites, "
                           "AA for Amino Acids and CoA for Coenzymes A")

        for col in ["%Diff", "Theoretical Amt", "Calculated Amt"]:
            qc_verif[col] = qc_verif[col].apply(lambda x: str(x).replace(",", "."))
            qc_verif[col] = pd.to_numeric(qc_verif[col], errors="coerce")

        self.qc_table = qc_verif[["Compound", "Theoretical Amt", "Calculated Amt", "%Diff"]]
        self.qc_table.set_index("Compound", inplace=True)
        if (abs(qc_verif["%Diff"].values) > 20).any():
            qc = False
        else:
            qc = True
        self.qc_table = self.qc_table.astype(str)

        self.qc_table = self.qc_table.style.apply(self._color_qc, axis=1, subset=["%Diff"])

        self.excel_tables.append(("Quality Control", self.qc_table))
        return qc

    def generate_areas_table(self):

        c12 = self.sample_data[~self.sample_data["Compound"].str.contains("C13")].copy()
        c13 = self.sample_data[self.sample_data["Compound"].str.contains("C13")].copy()

        c12_areas = pd.pivot_table(c12, "Area", "Compound", "Sample_Name")
        c13_areas = pd.pivot_table(c13, "Area", "Compound", "Sample_Name")

        self.c12_areas = c12_areas[~c12_areas.isin(["Excluded"]).all(axis=1)]
        self.c13_areas = c13_areas[~c13_areas.isin(["Excluded"]).all(axis=1)]

        self.excluded_c12_areas = c12_areas[c12_areas.isin(["Excluded"]).all(axis=1)]
        self.excluded_c13_areas = c13_areas[c13_areas.isin(["Excluded"]).all(axis=1)]
        if not self.excluded_c12_areas.empty or not self.excluded_c13_areas.empty:
            self.logger.info("\nSome metabolites were excluded during area table processing:\n")
            if not self.excluded_c12_areas.empty:
                self.logger.info(f"\n{self.excluded_c12_areas}")
            if not self.excluded_c12_areas.empty:
                self.logger.info(f"\n{self.excluded_c13_areas}")
        c12_cols = natsorted(self.c12_areas.columns)
        c13_cols = natsorted(self.c13_areas.columns)
        self.c12_areas = self.c12_areas[c12_cols]
        self.c13_areas = self.c13_areas[c13_cols]
        self.excel_tables.append(
            ("C12_areas", self.c12_areas)
        )
        self.excel_tables.append(
            ("C13_areas", self.c13_areas)
        )

    def _color_concentrations(self):
        pass

    def generate_concentrations_table(self, loq_export):

        concentrations = self.sample_data[~self.sample_data["Compound"].str.contains("C13")]
        concentrations = concentrations.pivot(index="Compound", columns="Sample_Name", values="Calculated Amt")
        concentrations, conc_nulls = self._replace(concentrations,
                                                   to_replace=[np.nan, 0],
                                                   value="NA", axis="row",
                                                   drop=True)
        self.concentration_table = concentrations.copy()
        self.loq_table = self._define_loq(concentrations)
        self.calib_data, cal_nulls = self._replace(self.calib_data,
                                                   to_replace=[np.nan, 0],
                                                   value="NA", axis="row",
                                                   drop=True)
        if not conc_nulls.empty:
            self.logger.info(f"\nRows removed from the concentration table:\n{conc_nulls.T}")
        if not cal_nulls.empty:
            self.logger.info(f"\nRows removed from the calibration table:\n{cal_nulls.T}")
        new_cols = natsorted(self.concentration_table.columns)
        self.concentration_table = self.concentration_table[new_cols]
        removed_loq = []
        for idx in self.loq_table.index:
            if (self.loq_table.loc[idx, :] == "<LLOQ").all() or (self.loq_table.loc[idx, :] == ">ULOQ").all():
                removed_loq.append(self.loq_table.loc[idx, :])
                self.loq_table.drop(idx, inplace=True)
        if removed_loq:
            self.logger.info(f"\nSome metabolite data are all outside the limit of quantification:"
                             f"\n{pd.concat(removed_loq, axis=1).T}\n")
        new_loq_cols = natsorted(self.loq_table.columns)
        self.loq_table = self.loq_table[new_loq_cols]
        self.excel_tables.append(
            ("Concentrations", self.concentration_table),
        )
        if loq_export:
            self.excel_tables.append(
                ("Concentrations_LLOQ", self.loq_table)
            )

    def get_ratios(self):

        # Isolate missing c13 compounds
        c12 = self.sample_data[~self.sample_data["Compound"].str.contains("C13")].copy()
        c13 = self.sample_data[self.sample_data["Compound"].str.contains("C13")].copy()
        c12_compounds, missing_c13_std = self._check_if_std(
            list(c12["Compound"].unique()), list(c13["Compound"].unique())
        )
        if missing_c13_std:
            self.logger.info(f"Metabolites missing from IDMS: \n{missing_c13_std}")
        else:
            self.logger.info("All mMetabolites are presentin the IDMS")

        # Drop missing compounds from c12 df
        c13.loc[:, "Compound"] = c13.loc[:, "Compound"].str.slice(0, -4)
        c12.set_index(["Compound", "Sample_Name"], inplace=True)
        c13.set_index(["Compound", "Sample_Name"], inplace=True)
        c12.sort_index(level=['Compound', 'Sample_Name'], inplace=True)
        c13.sort_index(level=['Compound', 'Sample_Name'], inplace=True)
        if missing_c13_std:
            c12.drop(missing_c13_std, inplace=True)

        self.logger.warning(f"\nMetabolites with null areas in c13 data:\n"
                            f"{pd.pivot_table(c13[c13['Area'] == 0], 'Area', 'Compound', 'Sample_Name')}\n")

        # Ensure that c12 and C13 have same indexes. Check both ways and isolate missing indexes. Compute ratios
        if c12.index.difference(c13.index).levshape != (0, 0) and c13.index.difference(c12.index).levshape != (0, 0):
            c12_diff = c12.index.difference(c13.index)
            c13_diff = c13.index.difference(c12.index)
            intercept = c12.index.intersection(c13.index)
            self.ratios = c12.loc[intercept, "Area"].divide(c13.loc[intercept, "Area"])
            self.ratios.name = "Ratios"
            self.no_ratio = {
                "c12": c12.loc[c12_diff, :],
                "c13": c13.loc[c13_diff, :]
            }
            self.logger.debug(
                f"Some index levels are in C12 data and not in C13 data. Differences:\n{self.no_ratio['c12']} "
                f"\n Some index levels are in C13 data and not in C12 data. Differences: \n{self.no_ratio['c13']}")
        else:
            if c12.index.difference(c13.index).levshape != (0, 0):
                c12_c13_diff = c12.index.difference(c13.index)
                self.ratios = c12.drop(c12_c13_diff).loc[c12_c13_diff, "Area"].divide(c13.loc[:, "Area"])
                self.ratios.name = "Ratios"
                self.no_ratio = c12.loc[c12_c13_diff, :]
                self.logger.info(f"Some index levels are in C12 data and not in C13 data. Differences:\n{c12_c13_diff}")
                print(f"Ratios calculated:\n{self.ratios}")
            elif c13.index.difference(c12.index).levshape != (0, 0):
                c13_c12_diff = c13.index.difference(c12.index)
                self.ratios = c12.loc[:, "Area"].divide(c13.drop(c13_c12_diff).loc[:, "Area"])
                self.ratios.name = "Ratios"
                self.no_ratio = c13.loc[c13_c12_diff, :]
                self.logger.info(f"Some index levels are in C13 data and not in C12 data. Differences:\n{c13_c12_diff}")
                print(f"Ratios calculated:\n{self.ratios}")
            else:
                self.ratios = c12.loc[:, "Area"].divide(c13.loc[:, "Area"])
                self.ratios.name = "Ratios"
                print(f"Ratios calculated with no differences detected between c12 and c13 indexes. "
                      f"Ratios:\n{self.ratios}")
        self.ratios = self.ratios.reset_index(level="Sample_Name")
        self.ratios = pd.pivot_table(self.ratios, "Ratios", "Compound", "Sample_Name")
        self.ratios, removed_ratios = self._replace(self.ratios, [0, np.inf, np.nan], "NA", "row", True)
        if not removed_ratios.empty:
            self.logger.info(f"\nSome rows were removed from the ratios dataframe because they contained only infinites,"
                             f"zeroes or NaNs.\nRemoved rows:\n{removed_ratios}")
        new_cols = natsorted(self.ratios.columns)
        self.ratios = self.ratios[new_cols]
        self.ratios = self.ratios.replace(r'^\s*$', "NA", regex=True)
        self.excel_tables.append(
            ("Ratios", self.ratios)
        )

    @staticmethod
    def _check_if_std(c12_compounds, c13_compounds):

        trunc_c13 = [std[:-4] for std in c13_compounds]
        missing_std = [x for x in c12_compounds if x not in trunc_c13]
        if missing_std:
            for x in missing_std:
                c12_compounds.remove(x)
            return c12_compounds, missing_std
        else:
            return c12_compounds, None

    @staticmethod
    def _isolate_nulls(df):

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

        for idx in loq_table.index:
            lloq_mask = loq_table.loc[idx, :].apply(lambda x: float(x) < self.calib_data.at[idx, "min"])
            uloq_mask = loq_table.loc[idx, :].apply(lambda x: float(x) > self.calib_data.at[idx, "max"])
            loq_table.loc[idx, :] = loq_table.loc[idx, :].where(~lloq_mask, other="<LLOQ")
            loq_table.loc[idx, :] = loq_table.loc[idx, :].where(~uloq_mask, other=">ULOQ")
        return loq_table

    def normalise(self, factor, unit):
        pass

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

    def generate_report(self):

        if self.calrep is None:
            raise ValueError("Cannot generate the report because no calibration report file was detected")
        report = self.calrep.iloc[10:, :8]
        report = report.iloc[:-2, [0, -1]].copy()
        report.columns = ["Compound", "R²"]
        report = report[~report["Compound"].str.contains("C13")]
        report = report.sort_values("Compound")
        report.set_index("Compound", inplace=True)
        self.calrep = report.copy()
        self.calrep["R²"] = self.calrep["R²"].astype(str)
        self.calrep = self.calrep.style.apply(Extractor.check_r, axis=1, subset=["R²"])

        self.excel_tables.append(
            ("Report", self.calrep)
        )

    def _output_log(self, destination):

        self.stream.seek(0)
        with open(f"{destination}\\file.log", "w") as log:
            print(self.stream.getvalue(), file=log)

    def export_stat_output(self, path, conc_unit):

        dest = Path(path)
        dest = dest / "output_for_graphstat.tsv"
        stat_out = self._build_stat_output(conc_unit)
        stat_out.to_csv(str(dest), sep="\t", index=False, encoding='utf-8-sig')
        self._output_log(path)

    def export_final_excel(self, path):

        dest_path = Path(path)
        dest_path = dest_path / "Tables.xlsx"
        with pd.ExcelWriter(str(dest_path)) as writer:
            for (name, table) in self.excel_tables:
                table.to_excel(writer, sheet_name=name, engine="openpyxl")
        self._output_log(str(path))
        print(f"Done exporting. Path:\n {str(dest_path)}")

    @staticmethod
    def _replace(
            df: pd.DataFrame,
            to_replace: int or str or float or list,
            value: Any,
            axis: str,
            drop: bool = False
    ) -> pd.DataFrame:

        """

        Homemade replace function :param df: dataframe in which to replace values :param to_replace: value(s) to
        replace :param value: value to replace with :param axis: axis on which to apply the method (can be whole
        dataframe) :param drop: should row or column be dropped if the replacing value take the whole axis (only for
        axis=row or column) :return: replaced dataframe and removed axes if drop=True

        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df} is not a dataframe, it is of type {type(df)}")
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
                                removed.append(df.loc[idx, :])
                                df = df.drop(idx)
                            else:
                                df.loc[idx, :] = value
                if axis == "column":
                    for col in df.columns:
                        if (df.loc[:, col].isin(to_replace)).all():
                            if drop:
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

    def _build_stat_output(self, conc_unit):

        to_out = []
        if isinstance(self.c12_areas, pd.DataFrame) and isinstance(self.c13_areas, pd.DataFrame):
            c12_areas = self._replace(self.c12_areas, [0, np.inf, ""], "NA", "row")
            c13_areas = self._replace(self.c13_areas, [0, np.inf, ""], "NA", "row")
            c12_areas = c12_areas.reset_index()
            c13_areas = c13_areas.reset_index()
            c12_areas = c12_areas.rename({"Compound": "Features"}, axis=1)
            c13_areas = c13_areas.rename({"Compound": "Features"}, axis=1)
            c12_areas.insert(1, "type", "C12 area")
            c12_areas.insert(2, "unit", "Arbitrary")
            c13_areas.insert(1, "type", "C13 area")
            c13_areas.insert(2, "unit", "Arbitrary")
            to_out.append(c12_areas)
            to_out.append(c13_areas)
        if isinstance(self.concentration_table, pd.DataFrame) or isinstance(self.loq_table, pd.DataFrame):
            concentrations = self._replace(self.loq_table.copy(), "<LLOQ", "NA", "dataframe")
            concentrations = self._replace(concentrations, ">ULOQ", "NA", "dataframe")
            concentrations = self._replace(concentrations, "", "NA", "dataframe")
            concentrations = concentrations.reset_index()
            concentrations = concentrations.rename({"Compound": "Features"}, axis=1)
            concentrations.insert(1, "type", "concentration")
            concentrations.insert(2, "unit", conc_unit)
            to_out.append(concentrations)
        if isinstance(self.ratios, pd.DataFrame):
            ratios = self.ratios.reset_index()
            ratios = self._replace(ratios, [np.inf, np.nan, ""], "NA", "dataframe")
            ratios = ratios.rename({"Compound": "Features"}, axis=1)
            ratios.insert(1, "type", "C12/C13 ratios")
            ratios.insert(2, "unit", "Arbitrary")
            to_out.append(ratios)
        stat_out = pd.concat(to_out)
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
    test = Extractor(r"C:\Users\legregam\Documents\Projets\MSReader\test\20210506_SOKOL_filtres_MC_quant.xlsx",
                     None, "AA")
    qc_result = test.handle_qc()

    # r"C:\Users\legregam\Documents\Projets\MSReader\test\Calibration Report.xlsx",
    # r"C:\Users\legregam\Documents\Projets\MSReader\test\Sample_List_test.xlsx",
    # test.handle_calibration()
    # test.generate_concentrations_table(True)
    # test.generate_report()
    # test.get_ratios()
