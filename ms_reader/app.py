import streamlit as st
import pandas as pd
from extract import Extractor

st.title("MS_Reader Demo App")

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

if data:
    if qc_type == "Amino Acids":
        raise NotImplementedError("Amino Acid QC not yet implemented")
    if qc_type == "Coenzymes A":
        raise NotImplementedError("Coenzyme A QC not yet implemented")
    data = pd.read_excel(data, engine="openpyxl")
    if report:
        report = pd.read_excel(report, engine="openpyxl")
    else:
        report = None
    if metadata:
        metadata = pd.read_excel(metadata, engine="openpyxl")
    else:
        report = None
    if qc_type == "Central Metabolites":
        qc_type = "CM"
    elif qc_type == "Amino Acids":
        qc_type = "AA"
    elif qc_type == "Coenzymes A":
        qc_type = "CoA"
    else:
        qc_type = None
    msr = Extractor(data, report, metadata, qc_type)
    if qc_type is not None:
        qc_result = msr.handle_qc()
        if not qc_result:
            st.error("QC not valid:")
            st.write(msr.qc_table)
        else:
            st.subheader("QC is valid:")
            st.write(msr.qc_table)
    st.subheader("Choose tables to output")
    with st.form("Table select"):
        cln1, cln2, cln3, cln4, cln5 = st.columns(5)
        with cln1:
            if msr.calrep is None:
                disable = True
            else:
                disable = False
            report_box = st.checkbox("Report", key="report_box", disabled=disable)
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
    msr.handle_calibration()
    if report_box:
        msr.generate_report()
        if preview:
            with st.expander("Show report"):
                st.dataframe(msr.calrep)
    if areas_box:
        msr.generate_areas_table()
        if preview:
            with st.expander("Show C12 Areas"):
                st.dataframe(msr.c12_areas)
                if not msr.excluded_c12_areas.empty:
                    st.write(f"Some metabolites were excluded:")
                    st.dataframe(msr.excluded_c12_areas)
            with st.expander("Show C13 Areas"):
                st.dataframe(msr.c13_areas)
                if not msr.excluded_c13_areas.empty:
                    st.write(f"Some metabolites were excluded:")
                    st.dataframe(msr.excluded_c13_areas)
    if ratios_box:
        msr.get_ratios()
        if preview:
            with st.expander("Show Ratios"):
                st.dataframe(msr.ratios)
    if conc_box or lloq_box:
        msr.generate_concentrations_table(lloq_box)
        if conc_box:
            if preview:
                with st.expander("Show concentrations (no lloq)"):
                    st.dataframe(msr.concentration_table.apply(
                        lambda x: x.astype(str)
                    ))
        if lloq_box:
            if preview:
                with st.expander("Show concentrations (with lloq)"):
                    st.dataframe(msr.loq_table.astype(str))
    if submit_export:
        msr.export_final_excel(destination)
        st.text("The final excel has been generated")
    if submit_stat_out:
        msr.export_stat_output(destination, concentration_unit)
        st.text("The output for the stat object has been generated")
