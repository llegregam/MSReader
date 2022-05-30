from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd

from extract import Extractor
from ms_reader import __version__, __file__


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
                f'You can update MS_reader with: "pip install --upgrade ms_reader". \n\n'
                f'Check the documentation for more information.'
            )
    except Exception:
        pass


@st.cache
def convert_df(df):
    """
    Convert dataframe to excel file stored in RAM and return it (for the download button widget)
    :param df: Dataframe containing data
    :return: Bytes containing the entire contents of the buffer
    """
    buffer = BytesIO()
    df.to_excel(buffer)
    return buffer.getvalue()


st.set_page_config(page_title=f"MS_Reader (v{__version__})")
st.title(f"Welcome to MS_Reader (v{__version__})")
check_uptodate()

col1, col2, col3 = st.columns(3)
with col1:
    data = st.file_uploader("Upload Data")
with col2:
    report = st.file_uploader("Upload Report File (optional)")
with col3:
    metadata = st.file_uploader("Upload Metadata (optional)")
qc_type = st.selectbox(
    "Choose molecular type",
    ["--", "Central Metabolites", "Amino Acids", "Coenzymes A"]
)
excel_engine = "openpyxl"
if data:

    # noinspection PyArgumentList
    data = pd.read_excel(io=data, engine=excel_engine)

    # Check if report and metadate files are given, if so read them
    if report:
        # noinspection PyArgumentList
        report = pd.read_excel(io=report, engine=excel_engine)
    else:
        report = None
    if metadata:
        # noinspection PyArgumentList
        metadata = pd.read_excel(io=metadata, engine=excel_engine)
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

    ms_reader = Extractor(data, report, metadata, qc_type)

    if metadata is None:
        number_norms = st.number_input(
            label="Normalisations",
            min_value=1,
            max_value=10,
            value=1,
            help="Select a number of normalisations columns for the metadata file"
        )

        st.download_button(
            label="Generate Metadata",
            data=convert_df(ms_reader.generate_metadata(number_norms)),
            file_name="Metadata.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Generate metadata with a number of normalisation columns equal to the number entered above"
        )

    if qc_type is not None:
        qc_result = ms_reader.handle_qc()
        if not qc_result:
            st.error("QC not valid:")
            st.write(ms_reader.qc_table)
        else:
            st.subheader("QC is valid:")
            st.write(ms_reader.qc_table)

    st.subheader("Choose tables to output")

    with st.form("Table select"):
        cln1, cln2, cln3, cln4, cln5 = st.columns(5)
        with cln1:
            report_box = st.checkbox(
                "Report", key="report_box",
                disabled=True if ms_reader.calrep is None else False
            )
        with cln2:
            areas_box = st.checkbox("Areas", key="areas_box")
        with cln3:
            ratios_box = st.checkbox("Ratios", key="ratios_box")
        with cln4:
            conc_box = st.checkbox("Concentrations", key="conc_box")
        with cln5:
            lloq_box = st.checkbox("LLoQ", key="lloq_box")

        concentration_unit = st.text_input("Input the concentration unit")
        destination = st.text_input("Input destination path for excel files")
        preview = st.form_submit_button("Preview")
        submit_export = st.form_submit_button("Export selection")
        submit_stat_out = st.form_submit_button("Export stat output")

    ms_reader.handle_calibration()

    if report_box:
        ms_reader.generate_report()
        if preview:
            with st.expander("Show report"):
                st.dataframe(ms_reader.calrep)
    if areas_box:
        ms_reader.generate_areas_table()
        if preview:
            with st.expander("Show C12 Areas"):
                st.dataframe(ms_reader.c12_areas)
                if not ms_reader.excluded_c12_areas.empty:
                    st.write(f"Some metabolites were excluded:")
                    st.dataframe(ms_reader.excluded_c12_areas)
            with st.expander("Show C13 Areas"):
                st.dataframe(ms_reader.c13_areas)
                if not ms_reader.excluded_c13_areas.empty:
                    st.write(f"Some metabolites were excluded:")
                    st.dataframe(ms_reader.excluded_c13_areas)
    if ratios_box:
        ms_reader.get_ratios()
        if preview:
            with st.expander("Show Ratios"):
                st.dataframe(ms_reader.ratios)
    if conc_box or lloq_box:
        ms_reader.generate_concentrations_table(lloq_box)
        if conc_box:
            if preview:
                with st.expander("Show concentrations (no lloq)"):
                    st.dataframe(ms_reader.concentration_table.apply(
                        lambda x: x.astype(str)
                    ))
        if lloq_box:
            if preview:
                with st.expander("Show concentrations (with lloq)"):
                    st.dataframe(ms_reader.loq_table.astype(str))
    if submit_export:
        ms_reader.export_final_excel(destination)
        st.text("The final excel has been generated")
    if submit_stat_out:
        ms_reader.export_stat_output(destination, concentration_unit)
        st.text("The output for the stat object has been generated")
