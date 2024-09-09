"""
Converter to convert the skyline input to MS_Reader format
"""

import re

import pandas as pd
import numpy as np

MAPPING = {
    "Molecule": "Compound",
    "File Name": "Filename",
    "Total Area": "Area",
    "Quantification": "Calculated Amt",
    "Explicit Analyte Concentration": "Theoretical Amt",
    "Exclude From Calibration": "Excluded",
    "Accuracy": "%Diff",
    "Normalized Area": "Response Ratio"
}

SAMPLE_TYPE_MAPPING = {
    "Quality Control": "QC Std",
    "Standard": "Cal Std",
    "Blank": "Unknown"
}


def convert_column_names(df):
    """
    This function is used to rename the columns of a DataFrame based on a predefined mapping.

    The mapping is defined in the MAPPING dictionary, where the keys are the original column names
    and the values are the new column names.

    :param df: The DataFrame whose columns are to be renamed.
    :return: The DataFrame with renamed columns.
    """
    return df.rename(MAPPING, axis=1)


def convert_sample_types(value):
    """
    This function is used to convert the sample types based on a predefined mapping.

    The mapping is defined in the SAMPLE_TYPE_MAPPING dictionary, where the keys are the original sample types
    and the values are the new sample types.

    :param value: The original sample type.

    :return: The new sample type if it exists in the mapping, otherwise returns the original value.
    """

    if value in SAMPLE_TYPE_MAPPING.keys():
        return SAMPLE_TYPE_MAPPING[value]
    return value


def convert_accuracy_to_diff(value):
    """
    This function is used to convert the accuracy value to a difference value.

    Parameters:
    value (str): The original accuracy value as a string with a '%' sign.

    Returns:
    float: The difference value as a float, rounded to 1 decimal place.
    """

    if value != np.nan:
        return round(float(str(value).replace("%", ""))-100, 1)


def convert_calculated_amt(value):
    """
    This function is used to convert the calculated amount value based on certain conditions.

    If the value is a string and contains either "Normalized Area" or "NaN", the function returns NaN.
    If the value is a string and does not contain either "Normalized Area" or "NaN", the function removes " uM"
    from the string and converts it to a float.

    Parameters:
    value (str): The original calculated amount value.

    Returns:
    float or NaN: The converted calculated amount value as a float, or NaN if the original value contained "Normalized Area" or "NaN".
    """

    if isinstance(value, str):
        if "Normalized Area" in value or "NaN" in value:
            return np.nan
        return float(str(value).replace(" uM", ""))


def handle_na(row):
    """
    This function is used to handle missing values in a given row of a DataFrame.

    The function checks for certain conditions in the row and modifies the values accordingly.
    If "(heavy)" is in the "Precursor" value, " C13" is appended to the "Compound" value.
    If the "Area" value is NaN, both the "Area" and "Calculated Amt" values are set to "N/F".
    If the "Calculated Amt" value is None, it is set to NaN.
    If both the "Area" and "Calculated Amt" values are not strings and the "Area" value is not NaN but the "Calculated Amt" value is NaN, the "Calculated Amt" value is set to NaN.
    If a TypeError occurs during the process, the row and the "Area" and "Calculated Amt" values are printed.

    Parameters:
    row (pandas.Series): The row of the DataFrame to handle missing values in.

    Returns:
    pandas.Series: The row of the DataFrame with handled missing values.
    """

    if "(heavy)" in row["Precursor"]:
        row["Compound"] += " C13"
    if np.isnan(row["Area"]):
        row["Area"] = row["Calculated Amt"] = "N/F"
    if row["Calculated Amt"] is None:
        row["Calculated Amt"] = np.nan
    if not isinstance(row["Area"], str) and not isinstance(row["Calculated Amt"], str):
        try:
            if not np.isnan(row["Area"]) and np.isnan(row["Calculated Amt"]):
                row["Calculated Amt"] = np.nan
        except TypeError:
            print(row)
            print(f"Area value = {row['Area']}\nCalculated Amt value = {row['Calculated Amt']}")
    return row


def import_skyline_dataset(skyline_file):
    """
    Import skyline dataset and transform into MS_Reader compatible format

    :param skyline_file: Bytes file containing skyline data (tabular format)
    """
    # Get copy of file binary to dodge any wierd effects when file is read twice by pandas
    # file = copy(skyline_file)
    filename_extension = skyline_file.name[-3:]
    if filename_extension not in ["tsv", "txt"]:
        raise TypeError(
            f"Skyline data must be in tabulated format with 'tsv' or 'txt' extension. "
            f"Detected extension: {filename_extension}"
        )

    data = pd.read_csv(skyline_file, sep="\t")
    # if len(data.columns) == 1:
    #    data = pd.read_csv(file, sep="\t")
    data = convert_column_names(data)
    data["Sample Type"] = data["Sample Type"].apply(convert_sample_types)
    data["%Diff"] = data["%Diff"].apply(convert_accuracy_to_diff).fillna("N/A")
    try:
        data["Calculated Amt"] = data["Calculated Amt"].apply(convert_calculated_amt)
    except ValueError:
        data["Calculated Amt"].replace(to_replace=re.compile(pattern='∞'), value="NaN", inplace=True)
        data["Calculated Amt"] = data["Calculated Amt"].apply(convert_calculated_amt)
    data = data.apply(handle_na, axis=1)

    return data


# if __name__ == "__main__":

#     with open(r"C:\Users\kouakou\Documents\MSREADER\data\20240715_GUILLOT_HILIC-POSNEG_QUANT_sansAA-NEG.tsv", "rb") as file:
#         data = import_skyline_dataset(file)
#         print(data["Response Ratio"])
#         # data.to_excel(r"C:\Users\kouakou\Documents\MSREADER\data\test2.xlsx")
