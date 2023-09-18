"""
Converter to convert the skyline input to MS_Reader format
"""

import pandas as pd
import numpy as np

MAPPING = {
    "Molecule": "Compound",
    "File Name": "Filename",
    "Total Area": "Area",
    "Quantification": "Calculated Amt",
    "Explicit Analyte Concentration": "Theoretical Amt",
    "Exclude From Calibration": "Excluded",
    "Accuracy": "%Diff"
}

SAMPLE_TYPE_MAPPING = {
    "Quality Control": "QC Std",
    "Standard": "Cal Std",
    "Blank": "Unknown"
}

def convert_column_names(df):

    return df.rename(MAPPING, axis=1)

def convert_sample_types(value):

    if value in SAMPLE_TYPE_MAPPING.keys():
        return SAMPLE_TYPE_MAPPING[value]
    return value

def convert_accuracy_to_diff(value):

    if value != np.nan:
        return round(float(str(value).replace("%", ""))-100, 1)

def convert_calculated_amt(value):

    if type(value) == str:
        if "Normalized Area" in value:
            return np.nan
        if "NaN" in value:
            return np.nan
        return float(str(value).replace(" uM", ""))

def handle_na(row):
    if "(heavy)" in row["Precursor"]:
        row["Compound"] = row["Compound"] + " C13"
    if np.isnan(row["Area"]):
        row["Area"] = "N/F"
        row["Calculated Amt"] = "N/F"
    if row["Calculated Amt"] is None:
        row["Calculated Amt"] = np.nan
    if not type(row["Area"]) == str and not type(row["Calculated Amt"]) == str:
        try:
            if not np.isnan(row["Area"]) and np.isnan(row["Calculated Amt"]):
                row["Calculated Amt"] = np.nan
        except TypeError:
            print(row)
            print(f"Area value = {row['Area']}\nCalculated Amt value = {row['Calculated Amt']}")
    return row

def convert_skyline_input(skyline_file):
    try:
        data = pd.read_csv(skyline_file, sep=",")
        data = convert_column_names(data)
        data["Sample Type"] = data["Sample Type"].apply(convert_sample_types)
        data["%Diff"] = data["%Diff"].apply(convert_accuracy_to_diff).fillna("N/A")
        data["Calculated Amt"] = data["Calculated Amt"].apply(convert_calculated_amt)
        data = data.apply(handle_na, axis=1)
    except Exception:
        raise
    return data

if __name__ == "__main__":

    #df = pd.read_csv(,sep=",")
    #print(df.columns)
    #converted_df = convert_column_names(df)
    #print(converted_df.columns)
    #print(converted_df["Sample Type"].unique())
    #converted_df["Sample Type"] = converted_df["Sample Type"].apply(convert_sample_types)
    #print(converted_df["Sample Type"].unique())
    #print(converted_df["%Diff"])
    #converted_df["%Diff"] = converted_df["%Diff"].apply(convert_accuracy_to_diff).fillna("NA")
    #print(converted_df["%Diff"])
    #converted_df["Calculated Amt"] = converted_df["Calculated Amt"].apply(convert_calculated_amt)
    #converted_df = converted_df.apply(handle_na, axis=1)
    #converted_df.to_excel(r"C:\Users\legregam\Desktop\test\test.xlsx", index=False)
    data = convert_skyline_input(r"C:\Users\legregam\PycharmProjects\MSReader\tests\data\skyline\Quantif-MC.csv")
    data.to_excel(r"C:\Users\legregam\Desktop\test\test2.xlsx")
