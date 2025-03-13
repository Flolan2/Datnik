import os
import pandas as pd
import numpy as np

def load_dat_scan(input_folder):
    """
    Load and preprocess the DatScan.csv file.
    Renames columns, converts the date column, and drops the Gender column.
    """
    dat_scan_path = os.path.join(input_folder, 'DatScan.csv')
    try:
        dat_scan_df = pd.read_csv(dat_scan_path, delimiter=';')
    except Exception as e:
        raise Exception(f"Error reading DatScan.csv: {e}")
    
    # Rename columns for clarity.
    rename_dict = {
        "No.": "Patient ID",
        "Date of Scan (DOS)": "Date of Scan",
        "Gender       0 = female\n1= male": "Gender",
        
        "Striatum Right: old software": "Striatum_Right_old",
        "Striatum Left: old software": "Striatum_Left_old",
        "Putamen Right: old software": "Putamen_Right_old",
        "Putamen Left: old software": "Putamen_Left_old",
        "Caudate Right: old software": "Caudate_Right_old",
        "Caudate Left: old software": "Caudate_Left_old",
        "Mean striatum: old software": "Mean_striatum_old",
        "Mean Putamen: old software": "Mean_Putamen_old",
        "Mean Caudate: old software": "Mean_Caudate_old",
        
        "Striatum Right: Z-Werte (new software)": "Striatum_Right_Z",
        "Striatum Left: Z-Werte (new software)": "Striatum_Left_Z",
        "Putamen Right: Z-Werte (new software)": "Putamen_Right_Z",
        "Putamen Left: Z-Werte (new software)": "Putamen_Left_Z",
        "Caudate Right: Z-Werte (new software)": "Caudate_Right_Z",
        "Caudate Left: Z-Werte (new software)": "Caudate_Left_Z",
        "Mean striatum: Z-Werte (new software)": "Mean_striatum_Z",
        "Mean Putamen: Z-Werte (new software)": "Mean_Putamen_Z",
        "Mean Caudate: Z-Werte (new software)": "Mean_Caudate_Z",
        
        "Striatum Right: new software": "Striatum_Right_new",
        "Striatum Leftt: new software": "Striatum_Left_new",
        "Putamen Rightt: new software": "Putamen_Right_new",
        "Putamen Leftt: new software": "Putamen_Left_new",
        "Caudate Rightt: new software": "Caudate_Right_new",
        "Caudate Leftt: new software": "Caudate_Left_new",
        "Mean striatumt: new software": "Mean_striatum_new",
        "Mean Putament: new software": "Mean_Putamen_new",
        "Mean Caudatet: new software": "Mean_Caudate_new"
    }
    dat_scan_df.rename(columns=rename_dict, inplace=True)
    
    # Convert 'Date of Scan' to datetime.
    dat_scan_df['Date of Scan'] = pd.to_datetime(dat_scan_df['Date of Scan'], errors='coerce', dayfirst=True)
    
    # Drop the Gender column as it is not needed.
    if 'Gender' in dat_scan_df.columns:
        dat_scan_df.drop(columns=['Gender'], inplace=True)
    
    # Debug print: Show all patient IDs loaded from DatScan.csv.
    print("Debug: Loaded DatScan.csv. Patient IDs:", dat_scan_df['Patient ID'].tolist())
    
    return dat_scan_df

def load_summary(input_folder):
    """
    Load the summary.csv file containing kinematic data.
    """
    summary_csv_path = os.path.join(input_folder, 'summary.csv')
    try:
        summary_df = pd.read_csv(summary_csv_path)
    except Exception as e:
        raise Exception(f"Error reading summary.csv: {e}")
    
    # Debug print: Show all patient IDs loaded from summary.csv.
    print("Debug: Loaded summary.csv. Patient IDs:", summary_df['Patient ID'].tolist())
    
    return summary_df

def merge_data(summary_df, dat_scan_df):
    """
    Merge the summary and DatScan data on 'Patient ID'.
    """
    print("Debug: Before merge - Summary patient IDs:", summary_df['Patient ID'].tolist())
    print("Debug: Before merge - DatScan patient IDs:", dat_scan_df['Patient ID'].tolist())
    
    merged_df = pd.merge(summary_df, dat_scan_df, on="Patient ID", how="inner")
    
    print("Debug: After merge - Merged patient IDs:", merged_df['Patient ID'].tolist())
    
    return merged_df

def convert_imaging_columns(merged_df):
    """
    Convert imaging columns from strings with commas to numeric values.
    """
    imaging_columns_old = [
        "Striatum_Right_old", "Striatum_Left_old", 
        "Putamen_Right_old", "Putamen_Left_old", 
        "Caudate_Right_old", "Caudate_Left_old",
        "Mean_striatum_old", "Mean_Putamen_old", "Mean_Caudate_old"
    ]
    imaging_columns_new = [
        "Striatum_Right_new", "Striatum_Left_new", 
        "Putamen_Right_new", "Putamen_Left_new", 
        "Caudate_Right_new", "Caudate_Left_new",
        "Mean_striatum_new", "Mean_Putamen_new", "Mean_Caudate_new"
    ]
    imaging_columns_z = [
        "Striatum_Right_Z", "Striatum_Left_Z", 
        "Putamen_Right_Z", "Putamen_Left_Z", 
        "Caudate_Right_Z", "Caudate_Left_Z",
        "Mean_striatum_Z", "Mean_Putamen_Z", "Mean_Caudate_Z"
    ]
    all_imaging_cols = imaging_columns_old + imaging_columns_new + imaging_columns_z

    for col in all_imaging_cols:
        if col in merged_df.columns:
            series = merged_df[col]
            # If the column returns a DataFrame (e.g. due to duplicate names), take the first column.
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            # Ensure that the series is a proper pandas Series.
            series = pd.Series(series)
            # Convert to string, replace commas with dots, then convert to numeric.
            series = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
            merged_df[col] = series
    
    return merged_df, all_imaging_cols, imaging_columns_old, imaging_columns_new, imaging_columns_z

def get_kinematic_variables(merged_df, all_imaging_cols):
    """
    Identify kinematic variables by excluding metadata and imaging columns.
    """
    metadata_cols = ['Patient ID', 'Date of Visit', 'Medication Condition', 'Kinematic Task', 'Hand Condition', 'Days Since First Visit']
    imaging_set = set(all_imaging_cols)
    kinematic_cols = [col for col in merged_df.columns if col not in metadata_cols and col not in imaging_set 
                      and pd.api.types.is_numeric_dtype(merged_df[col])]
    print("Debug: Kinematic variables identified:", kinematic_cols)
    return kinematic_cols

def get_patients_with_kinematic(merged_df, kinematic_cols):
    """
    Return a list of patient IDs that have at least one kinematic measurement.
    """
    patients_with_kinematic = merged_df.loc[merged_df[kinematic_cols].notna().any(axis=1), 'Patient ID']
    print("Debug: Patients with kinematic data:", patients_with_kinematic.tolist())
    return patients_with_kinematic

def reshape_imaging_data(merged_df, imaging_columns):
    """
    Reshape imaging data from wide to long format.
    
    For each imaging column that is lateralized (i.e. contains 'Left' or 'Right'), extract:
        - Anatomical Region (e.g., "Striatum", "Putamen", "Caudate", etc.)
        - Laterality (Left or Right)
        - Modality (old, new, or Z)
        - Imaging Value
    Returns a long-format DataFrame.
    """
    rows = []
    for col in imaging_columns:
        if ("Left" in col) or ("Right" in col):
            # Determine laterality.
            if "Left" in col:
                laterality = "Left"
            elif "Right" in col:
                laterality = "Right"
            else:
                laterality = None

            # Determine modality.
            if col.endswith("_old"):
                modality = "old"
            elif col.endswith("_new"):
                modality = "new"
            elif col.endswith("_Z"):
                modality = "Z"
            else:
                modality = "other"

            # Extract anatomical region by removing laterality and modality parts.
            # Example: "Striatum_Right_old" becomes "Striatum".
            region = col.replace("Right", "").replace("Left", "")
            region = region.replace("_old", "").replace("_new", "").replace("_Z", "")
            region = region.strip().replace("_", " ")
            
            # Build a DataFrame for this column.
            temp_df = pd.DataFrame({
                "Patient ID": merged_df["Patient ID"],
                "Anatomical Region": region,
                "Laterality": laterality,
                "Modality": modality,
                "Imaging Value": merged_df[col]
            })
            rows.append(temp_df)
    if rows:
        imaging_long_df = pd.concat(rows, ignore_index=True)
    else:
        imaging_long_df = pd.DataFrame()
    print("Debug: Reshaped imaging data into long format with shape:", imaging_long_df.shape)
    return imaging_long_df

def reshape_kinematic_data(merged_df, kinematic_cols):
    """
    Reshape kinematic data from wide to long format using the "Hand Condition" column
    for laterality. For each kinematic column, this function creates rows with:
        - Kinematic Variable (using the column name)
        - Laterality (from the "Hand Condition" column)
        - Kinematic Value (the measurement)
    Returns a long-format DataFrame.
    """
    rows = []
    for col in kinematic_cols:
        # Use the kinematic column name as the variable name.
        variable = col.strip().replace("_", " ")
        
        # Use the "Hand Condition" column for laterality.
        temp_df = pd.DataFrame({
            "Patient ID": merged_df["Patient ID"],
            "Kinematic Variable": variable,
            "Laterality": merged_df["Hand Condition"],
            "Kinematic Value": merged_df[col]
        })
        rows.append(temp_df)
    if rows:
        kinematic_long_df = pd.concat(rows, ignore_index=True)
    else:
        kinematic_long_df = pd.DataFrame()
    print("Debug: Reshaped kinematic data into long format with shape:", kinematic_long_df.shape)
    return kinematic_long_df

def preprocess(input_folder):
    """
    Run the full preprocessing pipeline.
    
    This version loads the data, merges it, converts imaging columns to numeric,
    performs outlier removal, and then reshapes both imaging and kinematic data into long format.
    """
    dat_scan_df = load_dat_scan(input_folder)
    summary_df = load_summary(input_folder)
    merged_df = merge_data(summary_df, dat_scan_df)
    merged_df, all_imaging_cols, imaging_columns_old, imaging_columns_new, imaging_columns_z = convert_imaging_columns(merged_df)
    kinematic_cols = get_kinematic_variables(merged_df, all_imaging_cols)
    patients_with_kinematic = get_patients_with_kinematic(merged_df, kinematic_cols)
    
    # ---- Outlier Removal Step ----
    numeric_columns = [col for col in all_imaging_cols if col in merged_df.columns] + kinematic_cols
    for col in numeric_columns:
        lower = merged_df[col].quantile(0.01)
        upper = merged_df[col].quantile(0.99)
        merged_df[col] = merged_df[col].where((merged_df[col] >= lower) & (merged_df[col] <= upper))
    print("Debug: Outlier removal complete. Merged data shape remains:", merged_df.shape)
    # ---- End Outlier Removal ----
    
    # Reshape imaging and kinematic data into long format.
    imaging_long_df = reshape_imaging_data(merged_df, all_imaging_cols)
    kinematic_long_df = reshape_kinematic_data(merged_df, kinematic_cols)
    
    return {
        'merged_df': merged_df,
        'all_imaging_cols': all_imaging_cols,
        'imaging_columns_old': imaging_columns_old,
        'imaging_columns_new': imaging_columns_new,
        'imaging_columns_z': imaging_columns_z,
        'kinematic_cols': kinematic_cols,
        'patients_with_kinematic': patients_with_kinematic,
        'imaging_long': imaging_long_df,
        'kinematic_long': kinematic_long_df
    }

def export_long_csv(input_folder, output_file="long_data.csv"):
    """
    Process the data and export a long-style CSV that combines both imaging and kinematic data.
    For imaging data, the 'Anatomical Region' and 'Laterality' columns are combined (e.g., 'Striatum_Left').
    For kinematic data, the 'Kinematic Variable' and 'Laterality' columns are combined (e.g., 'Speed_Left').
    An additional column 'Data Type' indicates whether the row is from imaging or kinematic data.
    """
    processed = preprocess(input_folder)
    imaging_long = processed['imaging_long']
    kinematic_long = processed['kinematic_long']
    
    # Prepare imaging data: combine region and laterality into one variable name.
    if not imaging_long.empty:
        imaging_long["Variable"] = imaging_long["Anatomical Region"].str.replace(" ", "") + "_" + imaging_long["Laterality"].str.lower()
        imaging_long = imaging_long.rename(columns={"Imaging Value": "Value"})
        imaging_long["Data Type"] = "Imaging"
        imaging_long = imaging_long[["Patient ID", "Variable", "Value", "Data Type"]]
    
    # Prepare kinematic data: combine kinematic variable and laterality.
    if not kinematic_long.empty:
        kinematic_long["Variable"] = kinematic_long["Kinematic Variable"].str.replace(" ", "") + "_" + kinematic_long["Laterality"].str.lower()
        kinematic_long = kinematic_long.rename(columns={"Kinematic Value": "Value"})
        kinematic_long["Data Type"] = "Kinematic"
        kinematic_long = kinematic_long[["Patient ID", "Variable", "Value", "Data Type"]]
    
    # Combine both long DataFrames.
    combined_long = pd.concat([imaging_long, kinematic_long], ignore_index=True)
    
    # Export to CSV.
    combined_long.to_csv(output_file, index=False)
    print(f"Long style CSV saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess imaging and kinematic data and export a long-style CSV."
    )
    # Default value set to "input_folder"
    parser.add_argument(
        "input_folder",
        nargs="?",
        default="input_folder",
        help="Path to the folder containing DatScan.csv and summary.csv (default: input_folder)"
    )
    parser.add_argument(
        "--output_file",
        default="long_data.csv",
        help="Name (and path) of the output CSV file (default: long_data.csv)"
    )
    args = parser.parse_args()

    export_long_csv(args.input_folder, args.output_file)
