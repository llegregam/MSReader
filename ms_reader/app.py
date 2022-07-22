from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd

from extract import Extractor
from ms_reader import __version__, __file__

# Constants
EXCEL_ENGINE = "openpyxl"
MIME = "application/vnd.openxmlformats-" \
       "officedocument.spreadsheetml.sheet"


def check_uptodate():
    """Compare installed and most recent package versions."""
    try:
        pf_path = Path(__file__).parent
        with open(str(Path(pf_path, "last_version.txt")), "r") as f:
            lastversion = f.read()
        if lastversion != __version__:
            # change the next line to streamlit
            st.info(
                f'New version available ({lastversion}). \n\n'
                f'You can update MS_reader with: '
                f'"pip install --upgrade ms_reader". \n\n'
                f'Check the documentation for more information.'
            )
    except Exception:
        pass


def df_format(x):
    """
    Function to apply str format to df for display in the GUI

    :param x: value to convert
    :return: converted value
    """
    return x.astype(str)


@st.cache
def convert_df(df):
    """
    Convert dataframe to excel file stored in RAM and return it
    (for the download button widget)

    :param df: Dataframe containing data
    :return: Bytes containing the entire contents of the buffer
    """
    buffer = BytesIO()
    df.to_excel(buffer)
    return buffer.getvalue()


st.set_page_config(page_title=f"MS_Reader (v{__version__})")
st.title(f"Welcome to MS_Reader (v{__version__})")
check_uptodate()

st.subheader("Select input files")
col1, col2, col3 = st.columns(3)
with col1:
    data = st.file_uploader("Upload Data")
with col2:
    report = st.file_uploader("Upload Report File (optional)")
with col3:
    metadata = st.file_uploader("Upload Metadata (optional)")

if data:

    # noinspection PyArgumentList
    data = pd.read_excel(io=data, engine=EXCEL_ENGINE)

    # Add way to drop metabolites from data
    with st.expander("Click to open metabolite remover"):
        with st.form("metabolite_dropper"):
            metabolites_to_drop = st.multiselect(
                label="Select metabolites to remove from the data",
                options=data["Compound"].unique(),
                help="The metabolites selected here will be removed from the "
                     "final generated data. To avoid any errors, it is "
                     "advised to remove both the 12C and 13C version of the "
                     "metabolite."
            )
            st.form_submit_button("Remove selected")
        if metabolites_to_drop:
            data = data.set_index("Compound").drop(
                metabolites_to_drop).reset_index()
        st.dataframe(df_format(data))

    qc_type = st.selectbox(
        "Choose molecular type (for quality control)",
        ["--", "Central Metabolites", "Amino Acids", "Coenzymes A"]
    )

    # Check if report and metadate files are given, if so read them
    if report:
        # noinspection PyArgumentList
        report = pd.read_excel(io=report, engine=EXCEL_ENGINE)
    else:
        report = None
    if metadata:
        # noinspection PyArgumentList
        metadata = pd.read_excel(io=metadata, engine=EXCEL_ENGINE)
    else:
        metadata = None

    # Get the metabolite class
    if qc_type == "Central Metabolites":
        qc_type = "CM"
    elif qc_type == "Amino Acids":
        qc_type = "AA"
    elif qc_type == "Coenzymes A":
        qc_type = "CoA"
    else:
        qc_type = None

    reader = Extractor(data, report, metadata, qc_type)

    if qc_type is not None:
        qc_result = reader.handle_qc()
        if not qc_result:
            st.error("QC not valid:")
            st.write(reader.qc_table)
        else:
            st.subheader("QC is valid:")
            st.write(reader.qc_table)

    if metadata is None:
        number_norms = st.number_input(
            label="Normalisations",
            min_value=1,
            max_value=10,
            value=1,
            help="Select a number of normalisations "
                 "columns for the metadata file"
        )

        st.download_button(
            label="Generate Metadata",
            data=convert_df(reader.generate_metadata(number_norms)),
            file_name="Metadata.xlsx",
            mime=MIME,
            help="Generate metadata with a number of normalisation "
                 "columns equal to the number entered above"
        )
    else:
        with st.expander("Normalisation will be applied following "
                         "the given metadata file (click to display)"):
            st.dataframe(reader.metadata)

    st.subheader("Choose tables to output")

    with st.form("Table select"):
        cln1, cln2, cln3, cln4, cln5 = st.columns(5)
        with cln1:
            report_box = st.checkbox(
                "Report", key="report_box",
                disabled=True if reader.calrep is None else False
            )
        with cln2:
            areas_box = st.checkbox("Areas", key="areas_box")
        with cln3:
            ratios_box = st.checkbox("Ratios", key="ratios_box")
        with cln4:
            conc_box = st.checkbox("Concentrations", key="conc_box")
        with cln5:
            lloq_box = st.checkbox("LLOQ", key="lloq_box")

        concentration_unit = st.text_input(
            label="Input the concentration unit"
            if reader.metadata is None
            else "Input the quantity unit",
            value="µM" if reader.metadata is None else "µmol"
        )

        destination = st.text_input("Input destination path for excel files")
        preview = st.form_submit_button("Preview")
        submit_export = st.form_submit_button("Export selection")
        submit_stat_out = st.form_submit_button("Export stat output")

    reader.handle_calibration()

    if report_box:
        reader.generate_report(metabolites_to_drop)
        if preview:
            with st.expander("Show report"):
                st.dataframe(reader.calrep)
    if areas_box:
        reader.generate_areas_table()
        if preview:
            with st.expander("Show C12 Areas"):
                st.dataframe(reader.c12_areas.apply(df_format))
                if not reader.excluded_c12_areas.empty:
                    st.write("Some metabolites were excluded:")
                    st.dataframe(reader.excluded_c12_areas)
            with st.expander("Show C13 Areas"):
                st.dataframe(reader.c13_areas.apply(df_format))
                if not reader.excluded_c13_areas.empty:
                    st.write("Some metabolites were excluded:")
                    st.dataframe(reader.excluded_c13_areas)
            if reader.metadata is not None:
                with st.expander("Show normalised 12C Areas"):
                    st.dataframe(reader.norm_c12_areas.apply(df_format))
    if ratios_box:
        reader.generate_ratios()
        if preview:
            with st.expander("Show Ratios"):
                st.dataframe(reader.ratios.apply(df_format))
            if reader.metadata is not None:
                with st.expander("Show normalised ratios"):
                    st.dataframe(reader.normalised_ratios.apply(df_format))
    if conc_box or lloq_box:
        reader.generate_concentrations_table(lloq_box, concentration_unit)
        if conc_box:
            if preview:
                if reader.metadata is None:
                    with st.expander("Show concentrations (no lloq)"):
                        st.dataframe(
                            reader.concentration_table.apply(df_format)
                        )
                else:
                    with st.expander("Show quantities (no lloq)"):
                        st.dataframe(
                            reader.quantities.apply(df_format)
                        )
                    with st.expander("Show normalised quantities (no lloq)"):
                        st.dataframe(
                            reader.normalised_quantities.apply(df_format)
                        )
        if lloq_box:
            if preview:
                if reader.metadata is None:
                    with st.expander("Show concentrations (with lloq)"):
                        st.dataframe(reader.loq_table.astype(str))
                else:
                    with st.expander(
                            "Show normalised quantities (with lloq)"
                    ):
                        st.dataframe(reader.loq_table.apply(df_format))
    if submit_export:
        reader.export_final_excel(destination)
        st.success("The final excel has been generated")
    if submit_stat_out:
        reader.export_stat_output(destination)
        st.success("The output for the stat object has been generated")
