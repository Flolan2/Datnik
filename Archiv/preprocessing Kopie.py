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

def prepare_contralateral_data(merged_df, imaging_columns, kinematic_cols):
    """
    Prepares data pairs for contralateral correlation analysis between imaging and kinematic variables.
    
    For each imaging column that contains laterality information (e.g., "Right" or "Left"), 
    this function finds the corresponding kinematic columns on the opposite (contralateral) side.
    
    Parameters:
        merged_df (pd.DataFrame): The merged dataset from the preprocessing pipeline.
        imaging_columns (list): List of imaging columns (e.g., from old, new, and Z metrics).
        kinematic_cols (list): List of kinematic variable columns.
        
    Returns:
        dict: A dictionary where each key is a tuple (imaging_col, kinematic_col) and each value 
              is a DataFrame with 'Patient ID', the imaging variable, and the corresponding contralateral kinematic variable.
    """
    contralateral_pairs = {}
    
    for im_col in imaging_columns:
        # Determine the side in the imaging column.
        if "Right" in im_col:
            target_side = "Left"
        elif "Left" in im_col:
            target_side = "Right"
        else:
            continue  # Skip columns without clear lateral information
        
        # Look for kinematic columns that mention the contralateral side.
        for kin_col in kinematic_cols:
            if target_side in kin_col:
                # Create a paired DataFrame with non-null values.
                pair_df = merged_df[['Patient ID', im_col, kin_col]].dropna()
                if not pair_df.empty:
                    contralateral_pairs[(im_col, kin_col)] = pair_df
    
    print("Debug: Prepared contralateral pairs for analysis:", list(contralateral_pairs.keys()))
    return contralateral_pairs


def preprocess(input_folder):
    """
    Run the full preprocessing pipeline.
    """
    dat_scan_df = load_dat_scan(input_folder)
    summary_df = load_summary(input_folder)
    merged_df = merge_data(summary_df, dat_scan_df)
    merged_df, all_imaging_cols, imaging_columns_old, imaging_columns_new, imaging_columns_z = convert_imaging_columns(merged_df)
    kinematic_cols = get_kinematic_variables(merged_df, all_imaging_cols)
    patients_with_kinematic = get_patients_with_kinematic(merged_df, kinematic_cols)
    
    # ---- Outlier Removal Step ----
    # Only include numeric columns that exist in merged_df.
    numeric_columns = [col for col in all_imaging_cols if col in merged_df.columns] + kinematic_cols
    for col in numeric_columns:
        lower = merged_df[col].quantile(0.01)
        upper = merged_df[col].quantile(0.99)
        merged_df[col] = merged_df[col].where((merged_df[col] >= lower) & (merged_df[col] <= upper))
    print("Debug: Outlier removal complete. Merged data shape remains:", merged_df.shape)
    # ---- End Outlier Removal ----

    
    return {
        'merged_df': merged_df,
        'all_imaging_cols': all_imaging_cols,
        'imaging_columns_old': imaging_columns_old,
        'imaging_columns_new': imaging_columns_new,
        'imaging_columns_z': imaging_columns_z,
        'kinematic_cols': kinematic_cols,
        'patients_with_kinematic': patients_with_kinematic
    }
