import pandas as pd
import pytest

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
def mydata(script_location):
    mydata = pd.read_excel(script_location.join("data/test_data.xlsx"))
    return mydata


@pytest.fixture
def no_cal_test_data(script_location):
    no_cal_test_data = pd.read_excel(
        script_location.join("data/no_cal/test_data.xlsx")
    )
    return no_cal_test_data


@pytest.fixture
def report(script_location):
    report = pd.read_excel(
        script_location.join("data/Calibration Report.xlsx")
    )
    return report


@pytest.fixture
def metadata(script_location):
    metadata = pd.read_excel(script_location.join("data/Metadata.xlsx"))
    return metadata


@pytest.fixture
def msr(mydata, report, metadata):
    return Extractor(mydata, report, metadata)


class TestExtractor:

    def test_metadata_generator(self, msr):
        metadata = msr.generate_metadata(1)
        assert len(metadata.columns) == 4
        assert "Resuspension_Volume" in metadata.columns
        assert "Volume_Unit" in metadata.columns
        assert "Norm1" in metadata.columns
        assert "Norm1_Unit" in metadata.columns
        assert metadata.index.values.all() == msr.data[
            "Sample_Name"
        ].unique().all()
        assert "µL" in list(metadata["Volume_Unit"])

    def test_areas_no_norm(self, mydata, script_location):
        ms_reader = Extractor(mydata)
        ms_reader.handle_calibration()
        ms_reader.generate_areas_table()
        good_c12_areas_no_norm = pd.read_excel(
            script_location.join("data/good_tables_no_norm.xlsx"),
            sheet_name="C12_areas",
            index_col="Compound"
        )
        good_c13_areas_no_norm = pd.read_excel(
            script_location.join("data/good_tables_no_norm.xlsx"),
            sheet_name="C13_areas",
            index_col="Compound"
        )
        pd.testing.assert_frame_equal(
            good_c12_areas_no_norm,
            ms_reader.c12_areas,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_c13_areas_no_norm,
            ms_reader.c13_areas,
            check_names=False
        )

    def test_areas_norm(self, mydata, metadata, script_location):
        msr = Extractor(data=mydata, metadata=metadata)
        msr.handle_calibration()
        msr.generate_areas_table()
        good_c12_areas_norm = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="C12_areas",
            index_col="Compound"
        )
        good_c13_areas_norm = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="C13_areas",
            index_col="Compound"
        )
        pd.testing.assert_frame_equal(
            good_c12_areas_norm,
            msr.c12_areas,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_c13_areas_norm,
            msr.c13_areas,
            check_names=False
        )

    def test_ratios_no_norm(self, mydata, script_location):
        msr = Extractor(data=mydata)
        msr.handle_calibration()
        msr.generate_ratios()
        good_ratios_no_norm = pd.read_excel(
            script_location.join("data/good_tables_no_norm.xlsx"),
            sheet_name="Ratios",
            index_col="Compound"
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                good_ratios_no_norm,
                msr.ratios,
                check_names=False
            )

    def test_ratios_norm(self, mydata, metadata, script_location):
        msr = Extractor(data=mydata, metadata=metadata)
        msr.handle_calibration()
        msr.generate_ratios()
        good_ratios_norm = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="Normalised_Ratios",
            index_col="Compound"
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                good_ratios_norm,
                msr.ratios,
                check_names=False
            )

    def test_concentration_no_norm(self, mydata, script_location):
        msr = Extractor(data=mydata)
        msr.handle_calibration()
        msr.generate_concentrations_table(loq_export=True)
        good_concentration_no_norm = pd.read_excel(
            script_location.join("data/good_tables_no_norm.xlsx"),
            sheet_name="Concentrations",
            index_col="Compound"
        )
        good_loq_no_norm = pd.read_excel(
            script_location.join("data/good_tables_no_norm.xlsx"),
            sheet_name="Concentrations_LLOQ",
            index_col="Compound"
        )
        pd.testing.assert_frame_equal(
            good_concentration_no_norm,
            msr.concentration_table,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_loq_no_norm,
            msr.loq_table,
            check_names=False
        )

    def test_concentration_norm(self, mydata, metadata, script_location):
        msr = Extractor(data=mydata, metadata=metadata)
        msr.handle_calibration()
        msr.generate_concentrations_table(loq_export=True)
        good_quantities = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="Quantities",
            index_col="Compound"
        )
        good_quantities_norm = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="Normalised_Quantities",
            index_col="Compound"
        )
        good_loq_norm = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="Normalised_Quantities_LLOQ",
            index_col="Compound"
        )
        pd.testing.assert_frame_equal(
            good_quantities,
            msr.quantities,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_quantities_norm,
            msr.normalised_quantities,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_loq_norm,
            msr.loq_table,
            check_names=False
        )

    def test_qc(self, mydata, report, script_location):
        msr = Extractor(data=mydata, calrep=report)
        msr.handle_calibration()
        msr.handle_qc()
        good_report = pd.read_excel(
            script_location.join("data/good_tables_norm.xlsx"),
            sheet_name="Quality Control",
            index_col="Compound"
        )
        msr.qc_table.data = msr.qc_table.data.astype(float)
        pd.testing.assert_frame_equal(
            good_report,
            msr.qc_table.data,  # We need to access the styler object's data
            check_names=False
        )

    def test_no_cal(self, no_cal_test_data, script_location):
        ms_reader = Extractor(no_cal_test_data)
        ms_reader.generate_areas_table()
        ms_reader.generate_ratios()
        good_c12_areas = pd.read_excel(
            script_location.join("data/no_cal/Tables.xlsx"),
            sheet_name="C12_areas",
            index_col="Compound"
        )
        good_c13_areas = pd.read_excel(
            script_location.join("data/no_cal/Tables.xlsx"),
            sheet_name="C13_areas",
            index_col="Compound"
        )
        good_ratios = pd.read_excel(
            script_location.join("data/no_cal/Tables.xlsx"),
            sheet_name="Ratios",
            index_col="Compound"
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                good_ratios,
                ms_reader.ratios,
                check_names=False
            )
        pd.testing.assert_frame_equal(
            good_c12_areas,
            ms_reader.c12_areas,
            check_names=False
        )
        pd.testing.assert_frame_equal(
            good_c13_areas,
            ms_reader.c13_areas,
            check_names=False
        )
