import pandas as pd
import pytest

from ms_reader.skyline_convert import (
    convert_sample_types, convert_calculated_amt,
    convert_accuracy_to_diff, convert_column_names,
    handle_na
)
from ms_reader.extract import Extractor

pd.options.mode.chained_assignment = None


@pytest.fixture(scope="module")
def script_location(request):
    """
    Return the directory of the currently running test script.
    See: https://stackoverflow.com/questions/34504757/
         get-pytest-to-look-within-the-base-directory-of-the-testing-script
    for more info
    """

    # uses .join instead of .dirname so we get a LocalPath object instead of
    # a string. LocalPath.join calls normpath for us when joining the path
    return request.fspath.join('..')


@pytest.fixture
def tracefinder_data(script_location):
    tracefinder_data = pd.read_excel(script_location.join("data/test_data.xlsx"))
    return tracefinder_data


@pytest.fixture
def tracefinder_report(script_location):
    tracefinder_report = pd.read_excel(
        script_location.join("data/Calibration Report.xlsx")
    )
    return tracefinder_report


@pytest.fixture
def metadata(script_location):
    metadata = pd.read_excel(script_location.join("data/Metadata.xlsx"))
    return metadata


@pytest.fixture
def skyline_df(script_location):
    return pd.read_csv(script_location.join("./data/skyline/Quantif-MC.txt"), sep="\t")


@pytest.fixture
def tracefinder_msr(tracefinder_data, tracefinder_report, metadata):
    return Extractor(tracefinder_data, tracefinder_report, metadata)


class TestExtractor:

    SKYLINE_COLUMNS = [
        "Filename", "Sample Type", "Compound", 'Molecule List Name',
        'Precursor', 'Normalization Method', 'Area', 'RatioLightToHeavy',
        'Calculated Amt', 'Internal Standard Concentration', '%Diff',
        'Theoretical Amt', 'Excluded', 'R Squared'
    ]

    def test_convert_skyline_input_colums(self, skyline_df):

        # print(skyline_df.columns)
        converted_df = convert_column_names(skyline_df)
        # print(converted_df.columns)
        for col in self.SKYLINE_COLUMNS:
            assert col in converted_df.columns

    def test_skyline_sample_types(self, skyline_df):

        df = convert_column_names(skyline_df)
        df["Sample Type"] = df["Sample Type"].apply(convert_sample_types)
        assert "Quality Control" not in df["Sample Type"].unique()
        assert "Standard" not in df["Sample Type"].unique()
        assert "Blank" not in df["Sample Type"].unique()
        assert "QC Std" in df["Sample Type"].unique()
        assert "Cal Std" in df["Sample Type"].unique()
        assert "Unknown" in df["Sample Type"].unique()

    def test_convert_accuracy_to_diff(self, skyline_df):

        df = convert_column_names(skyline_df)
        df["Sample Type"] = df["Sample Type"].apply(convert_sample_types)
        df["%Diff"] = df["%Diff"].apply(convert_accuracy_to_diff).fillna("NA")
        assert "NA" in df["%Diff"].unique()
        for val in df["%Diff"].unique():
            assert "%" not in str(val)

    def test_convert_calculated_amt(self, skyline_df):
        df = convert_column_names(skyline_df)
        df["Sample Type"] = df["Sample Type"].apply(convert_sample_types)
        df["%Diff"] = df["%Diff"].apply(convert_accuracy_to_diff).fillna("NA")
        df["Calculated Amt"] = df["Calculated Amt"].apply(convert_calculated_amt)
        for val in df["Calculated Amt"].unique():
            assert isinstance(val, float)

    def test_handle_nans(self, skyline_df):
        df = convert_column_names(skyline_df)
        df["Sample Type"] = df["Sample Type"].apply(convert_sample_types)
        df["%Diff"] = df["%Diff"].apply(convert_accuracy_to_diff).fillna("NA")
        df["Calculated Amt"] = df["Calculated Amt"].apply(convert_calculated_amt)
        df = df.apply(handle_na, axis=1)
        assert "N/F" in df["Area"].unique()
        assert "N/F" in df["Calculated Amt"].unique()

    def test_metadata_generator(self, tracefinder_msr):
        metadata = tracefinder_msr.generate_metadata(1)
        assert len(metadata.columns) == 4
        assert "Resuspension_Volume" in metadata.columns
        assert "Volume_Unit" in metadata.columns
        assert "Norm1" in metadata.columns
        assert "Norm1_Unit" in metadata.columns
        assert metadata.index.values.all() == tracefinder_msr.data[
            "Sample_Name"
        ].unique().all()
        assert "µL" in list(metadata["Volume_Unit"])

    def test_areas_no_norm(self, tracefinder_data, script_location):
        ms_reader = Extractor(tracefinder_data)
        ms_reader.handle_calibration()
        ms_reader.generate_areas_table()
        assert not ms_reader.c12_areas.empty
        assert "N/F" not in ms_reader.c12_areas.values
        assert not ms_reader.c12_areas.empty
        assert "N/F" not in ms_reader.c13_areas.values
        assert "Excluded" in ms_reader.calib_data.values

    def test_areas_norm(self, tracefinder_data, metadata, script_location):
        msr = Extractor(data=tracefinder_data, metadata=metadata)
        msr.handle_calibration()
        msr.generate_areas_table()
        assert not msr.c12_areas.empty
        assert "N/F" not in msr.c12_areas.values
        assert not msr.c12_areas.empty
        assert "N/F" not in msr.c13_areas.values
        assert "Excluded" in msr.calib_data.values

    def test_ratios_no_norm(self, tracefinder_data, script_location):
        msr = Extractor(data=tracefinder_data)
        msr.handle_calibration()
        msr.generate_ratios()
        assert not msr.ratios.empty
        assert "NA" in msr.ratios.values

    def test_ratios_norm(self, tracefinder_data, metadata, script_location):
        msr = Extractor(data=tracefinder_data, metadata=metadata)
        msr.handle_calibration()
        msr.generate_ratios()
        assert not msr.ratios.empty
        assert "NA" in msr.ratios.values

    def test_concentration_no_norm(self, tracefinder_data, script_location):
        msr = Extractor(data=tracefinder_data)
        msr.handle_calibration()
        msr.generate_concentrations_table(loq_export=True)
        assert not msr.concentration_table.empty
        assert "<LLOQ" in msr.loq_table.values
        assert ">ULOQ" in msr.loq_table.values

    def test_concentration_norm(self, tracefinder_data, metadata, script_location):
        msr = Extractor(data=tracefinder_data, metadata=metadata)
        msr.handle_calibration()
        msr.generate_concentrations_table(loq_export=True)
        assert not msr.concentration_table.empty
        assert "<LLOQ" in msr.loq_table.values
        assert ">ULOQ" in msr.loq_table.values
