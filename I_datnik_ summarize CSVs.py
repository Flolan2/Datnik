# --- START OF FILE datnik_summarize_CSVs.py (MODIFIED to include Age data) ---

import os
import re
import pandas as pd
import csv
import numpy as np
from datetime import datetime

# Get the current script directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(script_dir) # Assumes script is in a subfolder like 'Online'
input_folder_path = os.path.join(project_root_dir, 'Input') # Input is sibling to script's parent
# Define an output data directory, sibling to 'Input' and 'Online'
output_data_dir = os.path.join(project_root_dir, 'Output', 'Data_Processed')
os.makedirs(output_data_dir, exist_ok=True) # Ensure it exists


def detect_separator(file_path):
    """Detects the CSV separator."""
    try:
        # Use utf-8-sig to handle potential BOM at the start of the file
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as file:
            sample = file.read(4096)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=',;')
                return dialect.delimiter
            except csv.Error:
                semicolon_count = sample.count(';')
                comma_count = sample.count(',')
                if semicolon_count > comma_count and comma_count == 0:
                    return ';'
                elif comma_count > semicolon_count and semicolon_count == 0:
                    return ','
                elif semicolon_count > 0: # Default to semicolon if mixed and it's present
                    return ';'
                else: # Default to comma if mixed and semicolon is not present, or if neither found clearly
                    return ','
    except Exception as e:
        print(f"Warning: Could not detect separator for {os.path.basename(file_path)} - {e}. Assuming ','.")
        return ','

def parse_kinematic_filename(filename):
    """
    Expected format: DateOfVisit_PatientID_MedCondition_KinematicTask - HandCondition.csv
    Returns: date_of_visit_str, patient_id, med_condition_standardized, kinematic_task_name, hand_condition_cleaned
    """
    base_name = os.path.splitext(filename)[0]
    base_name = base_name.replace('_thumbsize', '') # Remove specific suffix if present
    parts = base_name.split('_')

    if len(parts) < 4:
        print(f"Skipping file {filename}: unexpected filename format (too few underscores: {len(parts)-1})")
        return None

    date_of_visit_raw = parts[0]
    patient_id_raw = parts[1]
    med_condition_raw = parts[2]
    task_hand_part = "_".join(parts[3:])
    kinematic_task_name = ""
    hand_condition_cleaned = ""

    if ' - ' in task_hand_part:
        task_parts = task_hand_part.split(' - ', 1)
        kinematic_task_name = task_parts[0].strip()
        if len(task_parts) > 1:
            hand_condition_cleaned = task_parts[1].strip()
        else:
            print(f"Warning: Found ' - ' but no text after it for hand condition in {filename}.")
    else:
        kinematic_task_name = task_hand_part.strip()
        print(f"Warning: Could not find ' - ' separator for hand condition in {filename}.")

    try:
        date_of_visit_dt = pd.to_datetime(date_of_visit_raw, format='%Y-%m-%d')
    except ValueError:
        try:
            date_of_visit_dt = pd.to_datetime(date_of_visit_raw, format='%Y%m%d')
        except ValueError:
            print(f"Skipping file {filename}: could not parse date '{date_of_visit_raw}'.")
            return None
    date_of_visit_str = date_of_visit_dt.strftime('%d.%m.%Y')

    patient_id = str(patient_id_raw).strip()
    med_condition_standardized = med_condition_raw.strip().lower()
    if med_condition_standardized not in ['off', 'on']:
        print(f"Warning: Unexpected Medication Condition '{med_condition_raw}' in {filename}. Using '{med_condition_standardized}'.")

    if "left" in hand_condition_cleaned.lower(): hand_condition_cleaned = "Left"
    elif "right" in hand_condition_cleaned.lower(): hand_condition_cleaned = "Right"
    
    return date_of_visit_str, patient_id, med_condition_standardized, kinematic_task_name, hand_condition_cleaned


def process_kinematic_subfolder(base_folder_path, task_type_prefix):
    """
    Processes all CSV files in a specific base task folder AND its 'MedON' subfolder.
    Returns a DataFrame of raw kinematic performances.
    """
    raw_task_data_list = []
    medon_subfolder_path = os.path.join(base_folder_path, 'MedON')
    folders_to_scan = [base_folder_path]
    if os.path.isdir(medon_subfolder_path):
        folders_to_scan.append(medon_subfolder_path)
        print(f"\nFound 'MedON' subfolder for {task_type_prefix}: {medon_subfolder_path}")

    print(f"Processing task type '{task_type_prefix.upper()}' in folders: {folders_to_scan}")
    total_processed_files, total_skipped_files = 0, 0

    for current_folder in folders_to_scan:
        print(f"--- Scanning folder: {current_folder} ---")
        if not os.path.isdir(current_folder):
            print(f"Warning: Folder not found, skipping: {current_folder}"); continue
        processed_in_folder, skipped_in_folder = 0, 0
        for filename in os.listdir(current_folder):
            if not filename.endswith('.csv') or filename.startswith('.'): continue
            file_path = os.path.join(current_folder, filename)
            parsed = parse_kinematic_filename(filename)
            if parsed is None: skipped_in_folder += 1; continue
            date_str, pid, med_cond, task_name_from_file, hand_perf = parsed

            file_separator = detect_separator(file_path)
            try:
                header_to_use, names_to_use = 0, None
                try: # Check for 'attribute;value' style header
                    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        first_line = f.readline().strip()
                    expected_header_pat = re.compile(f"attribute\\s*{re.escape(file_separator)}\\s*value", re.IGNORECASE)
                    if not expected_header_pat.match(first_line):
                        header_to_use, names_to_use = None, ['Attribute', 'Value']
                except Exception as e_head:
                    print(f"Warning: Could not read first line of {filename} to check header: {e_head}. Assuming no header and ['Attribute', 'Value'].")
                    header_to_use, names_to_use = None, ['Attribute', 'Value']


                data_df = pd.read_csv(file_path, sep=file_separator, header=header_to_use, names=names_to_use, encoding='utf-8', on_bad_lines='warn')
                
                if 'Attribute' not in data_df.columns or 'Value' not in data_df.columns:
                    print(f"Skipping file {filename}: Missing 'Attribute' or 'Value' column after load. Columns found: {data_df.columns.tolist()}")
                    skipped_in_folder += 1; continue
                    
                data_df['Attribute'] = data_df['Attribute'].astype(str).str.strip().str.lower()
                data_df['Value'] = pd.to_numeric(data_df['Value'].astype(str).str.strip().str.replace(',', '.'), errors='coerce')
                data_df.dropna(subset=['Value'], inplace=True)
                if data_df.empty: skipped_in_folder += 1; continue
                
                data_df = data_df.drop_duplicates(subset=['Attribute'], keep='first')
                variables = data_df.set_index('Attribute')['Value'].to_dict()
                
                single_task_performance = {
                    'Patient ID': pid, 'Date of Visit': date_str, 'Medication Condition': med_cond,
                    'Hand_Performed': hand_perf, 'Task_Type': task_type_prefix,
                    'Original_Task_Name_From_File': task_name_from_file
                }
                single_task_performance.update(variables)
                raw_task_data_list.append(single_task_performance)
                processed_in_folder += 1
            except Exception as e_proc:
                print(f"Skipping file {filename}: could not process. Error: {repr(e_proc)}")
                skipped_in_folder += 1
        print(f"--- Finished folder scan: Processed: {processed_in_folder}, Skipped: {skipped_in_folder} ---")
        total_processed_files += processed_in_folder
        total_skipped_files += skipped_in_folder
    print(f"\nFinished processing task type '{task_type_prefix.upper()}': Total Processed {total_processed_files} files, Total Skipped {total_skipped_files} files.")
    return pd.DataFrame(raw_task_data_list)


def load_dat_scan(input_folder):
    dat_scan_path = os.path.join(input_folder, 'DatScan.csv')
    dat_scan_df = None
    print(f"\nAttempting to load DatScan data from: {dat_scan_path}")
    try:
        separator = detect_separator(dat_scan_path)
        dat_scan_df = pd.read_csv(dat_scan_path, delimiter=separator, encoding='utf-8-sig', on_bad_lines='warn')
        print(f"Successfully loaded DatScan.csv using '{separator}' separator.")
    except Exception as e_load:
        print(f"Error loading DatScan.csv: {repr(e_load)}")
        return None
    
    if dat_scan_df is None or dat_scan_df.empty:
        print("DatScan DataFrame could not be loaded or is empty.")
        return None

    current_cols_map_lower = {col.strip().lower(): col for col in dat_scan_df.columns}
    
    rename_dict_flexible = {
        "Patient ID": ["no.", "patient id", "patientid"],
        "Sex": ["gender", "sex"],
        "Date of Scan": ["date of scan (dos)", "date of scan", "scandate"],
        "Striatum_Right_Z": ["striatum right: z-werte (new software)", "striatum right z", "r striatum z-score"],
        "Striatum_Left_Z":  ["striatum left: z-werte (new software)", "striatum left z", "l striatum z-score"],
        "Putamen_Right_Z":  ["putamen right: z-werte (new software)", "putamen right z", "r putamen z-score"],
        "Putamen_Left_Z":   ["putamen left: z-werte (new software)", "putamen left z", "l putamen z-score"],
        "Caudate_Right_Z":  ["caudate right: z-werte (new software)", "caudate right z", "r caudate z-score"],
        "Caudate_Left_Z":   ["caudate left: z-werte (new software)", "caudate left z", "l caudate z-score"],
        "Striatum_Right_Raw": ["striatum right: new software", "striatum right", "r striatum sbr"],
        "Striatum_Left_Raw":  ["striatum leftt: new software", "striatum left: new software", "striatum left", "l striatum sbr"],
        "Putamen_Right_Raw":  ["putamen rightt: new software", "putamen right: new software", "putamen right", "r putamen sbr"],
        "Putamen_Left_Raw":   ["putamen leftt: new software", "putamen left: new software", "putamen left", "l putamen sbr"],
        "Caudate_Right_Raw":  ["caudate rightt: new software", "caudate right: new software", "caudate right", "r caudate sbr"],
        "Caudate_Left_Raw":   ["caudate leftt: new software", "caudate left: new software", "caudate left", "l caudate sbr"],
        "Mean_Striatum_Raw": ["mean striatumt: new software", "mean striatum: new software", "mean striatum"],
        "Mean_Putamen_Raw":  ["mean putament: new software", "mean putamen: new software", "mean putamen"],
        "Mean_Caudate_Raw":  ["mean caudatet: new software", "mean caudate: new software", "mean caudate"],
    }
    
    actual_rename_dict = {}
    all_target_columns = list(rename_dict_flexible.keys())
    
    for target_name, potential_sources in rename_dict_flexible.items():
        found_source = False
        for source_variant in potential_sources:
            source_variant_lower = source_variant.strip().lower()
            if source_variant_lower in current_cols_map_lower:
                actual_rename_dict[current_cols_map_lower[source_variant_lower]] = target_name
                found_source = True
                break 
        if not found_source and target_name not in ["Patient ID", "Date of Scan", "Mean_Striatum_Raw", "Mean_Putamen_Raw", "Mean_Caudate_Raw"]:
            print(f"Info: Critical DaTscan target column '{target_name}' not found in DatScan.csv source columns based on provided variants.")

    dat_scan_df.rename(columns=actual_rename_dict, inplace=True)
    
    essential_cols = ["Patient ID", "Date of Scan"]
    final_dat_scan_cols_to_keep = [col for col in essential_cols if col in dat_scan_df.columns]

    for target_col_name in all_target_columns:
        if target_col_name in dat_scan_df.columns and target_col_name not in final_dat_scan_cols_to_keep:
            final_dat_scan_cols_to_keep.append(target_col_name)
            
    if "Patient ID" not in dat_scan_df.columns:
        print("Error: 'Patient ID' column is missing in DatScan.csv after attempting to rename. Cannot proceed with DatScan processing.")
        return None
        
    dat_scan_df = dat_scan_df[final_dat_scan_cols_to_keep].copy()

    if "Date of Scan" in dat_scan_df.columns:
        dat_scan_df['Date of Scan'] = pd.to_datetime(dat_scan_df['Date of Scan'], errors='coerce', dayfirst=True)

    dat_scan_df.dropna(subset=['Patient ID'], inplace=True)
    dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(str).str.strip()
    try:
        numeric_pid = pd.to_numeric(dat_scan_df['Patient ID'], errors='coerce')
        dat_scan_df['Patient ID'] = np.where(numeric_pid.notna(), numeric_pid.astype(int).astype(str), dat_scan_df['Patient ID'])
    except Exception as e_pid_std:
        print(f"Warning: Could not fully standardize DatScan Patient IDs: {e_pid_std}")

    for col in dat_scan_df.columns:
        if '_Z' in col or '_Raw' in col:
            if col in dat_scan_df.columns:
                dat_scan_df[col] = pd.to_numeric(dat_scan_df[col].astype(str).str.replace(',', '.'), errors='coerce')
        # Convert numeric Sex/Gender (0/1) to categorical text
    if 'Sex' in dat_scan_df.columns:
        print("Info: Converting 'Sex' column from numeric (0/1) to categorical ('Female'/'Male').")
        sex_map = {0: 'Female', 1: 'Male'}
        # Ensure the column is numeric before mapping, handling potential string '0' or '1'
        dat_scan_df['Sex'] = pd.to_numeric(dat_scan_df['Sex'], errors='coerce')
        dat_scan_df['Sex'] = dat_scan_df['Sex'].map(sex_map)

            
    print(f"Processed DatScan data. Final shape: {dat_scan_df.shape}, Kept columns after processing: {dat_scan_df.columns.tolist()}")
    return dat_scan_df


def merge_dat_scan_hand_specific(kinematic_df, patient_info_df):
    if 'Patient ID' not in kinematic_df.columns or 'Hand_Performed' not in kinematic_df.columns:
        print("Warning: Kinematic DF missing 'Patient ID' or 'Hand_Performed'. Skipping merge logic.")
        return kinematic_df
    if patient_info_df is None or patient_info_df.empty:
        print("Warning: Patient Info DF is empty. Skipping merge logic.")
        return kinematic_df

    kinematic_df['Patient ID'] = kinematic_df['Patient ID'].astype(str).str.strip()
    try:
        numeric_pid_kin = pd.to_numeric(kinematic_df['Patient ID'], errors='coerce')
        kinematic_df['Patient ID'] = np.where(numeric_pid_kin.notna(), numeric_pid_kin.astype(int).astype(str), kinematic_df['Patient ID'])
    except Exception as e_pid_merge_kin:
        print(f"Warning during PID standardization for kinematic_df for merge: {e_pid_merge_kin}")

    merged_df = pd.merge(kinematic_df, patient_info_df, on="Patient ID", how="left", suffixes=('', '_InfoSuffix'))
    
    imaging_keys_base = ['Striatum', 'Putamen', 'Caudate']
    imaging_versions_to_process = ['Z', 'Raw']

    def select_ipsi_contra_hand_specific(row):
        hand = row.get('Hand_Performed')
        if pd.isna(hand): return row 
        hand_clean = str(hand).strip().lower()
        
        side_for_contralateral = None
        side_for_ipsilateral = None

        if hand_clean == 'left':
            side_for_contralateral = 'Right'
            side_for_ipsilateral = 'Left'
        elif hand_clean == 'right':
            side_for_contralateral = 'Left'
            side_for_ipsilateral = 'Right'
        else: 
            return row

        row_dict = row.to_dict()
        for key_base in imaging_keys_base:
            for version in imaging_versions_to_process:
                source_col_contra_name = f"{key_base}_{side_for_contralateral}_{version}"
                target_col_contra_name = f"Contralateral_{key_base}_{version}"
                
                value_contra = np.nan
                if source_col_contra_name in row_dict:
                    value_contra = row_dict.get(source_col_contra_name)
                row_dict[target_col_contra_name] = value_contra

                source_col_ipsi_name = f"{key_base}_{side_for_ipsilateral}_{version}"
                target_col_ipsi_name = f"Ipsilateral_{key_base}_{version}"
                
                value_ipsi = np.nan
                if source_col_ipsi_name in row_dict:
                    value_ipsi = row_dict.get(source_col_ipsi_name)
                row_dict[target_col_ipsi_name] = value_ipsi
        
        return pd.Series(row_dict)

    merged_df = merged_df.apply(select_ipsi_contra_hand_specific, axis=1)
    
    cols_to_drop_original_datscan_sides = []
    for key_base in imaging_keys_base:
        for version in imaging_versions_to_process: 
            for side in ['Left', 'Right']:
                original_side_specific_col = f"{key_base}_{side}_{version}"
                cols_to_drop_original_datscan_sides.append(original_side_specific_col)
                cols_to_drop_original_datscan_sides.append(f"{original_side_specific_col}_InfoSuffix")
    
    cols_to_drop_existing = [col for col in cols_to_drop_original_datscan_sides if col in merged_df.columns]
    if cols_to_drop_existing:
        merged_df.drop(columns=cols_to_drop_existing, inplace=True, errors='ignore')
        print(f"Dropped original L/R DatScan columns (Z and Raw): {cols_to_drop_existing}")
        
    return merged_df


def load_age_data(input_folder):
    """Loads the Age.csv file, cleans it, and prepares it for merging."""
    age_file_path = os.path.join(input_folder, 'Age.csv')
    print(f"\nAttempting to load Age data from: {age_file_path}")
    
    try:
        # Use encoding='utf-8-sig' to handle potential BOM characters at the start of the file
        age_df = pd.read_csv(age_file_path, sep=';', encoding='utf-8-sig', on_bad_lines='warn')
        
        # --- Data Cleaning and Standardization ---
        # Rename 'No.' to 'Patient ID' to match other dataframes
        if 'No.' in age_df.columns:
            age_df.rename(columns={'No.': 'Patient ID'}, inplace=True)
        else:
            print("Warning: 'No.' column not found in Age.csv. Cannot merge age data.")
            return None

        # Keep only the essential columns
        if 'Age' in age_df.columns and 'Patient ID' in age_df.columns:
            age_df = age_df[['Patient ID', 'Age']].copy()
        else:
            print("Warning: 'Patient ID' or 'Age' column missing after rename. Cannot merge age data.")
            return None
            
        # Drop rows where essential data is missing (handles empty lines at the end)
        age_df.dropna(subset=['Patient ID', 'Age'], inplace=True)
        
        # Standardize 'Patient ID' to be a string, consistent with other dataframes
        age_df['Patient ID'] = age_df['Patient ID'].astype(str).str.strip()
        try:
            # Replicate the exact PID standardization from other functions for a perfect merge
            numeric_pid = pd.to_numeric(age_df['Patient ID'], errors='coerce')
            age_df['Patient ID'] = np.where(numeric_pid.notna(), numeric_pid.astype(int).astype(str), age_df['Patient ID'])
        except Exception as e_pid_std_age:
            print(f"Warning: Could not fully standardize Age.csv Patient IDs: {e_pid_std_age}")
        
        print(f"Successfully loaded and processed Age data. Shape: {age_df.shape}")
        return age_df
        
    except FileNotFoundError:
        print(f"Info: Age.csv not found at {age_file_path}. Proceeding without age data.")
        return None
    except Exception as e:
        print(f"Error loading or processing Age.csv: {repr(e)}")
        return None


# ========================
# Main Execution Block
# ========================
if __name__ == '__main__':
    print("--- Starting Preprocessing Script (Hand-Centric, Ipsi/Contra Raw&Z, Exclusions, New Filename, with Age) ---")
    fingertapping_path = os.path.join(input_folder_path, 'Fingertapping')
    hand_movements_path = os.path.join(input_folder_path, 'Hand_Movements')

    ft_raw_df = process_kinematic_subfolder(fingertapping_path, "ft")
    hm_raw_df = process_kinematic_subfolder(hand_movements_path, "hm")

    kinematic_summary_df = pd.DataFrame()
    if not ft_raw_df.empty or not hm_raw_df.empty:
        all_tasks_list = [df for df in [ft_raw_df, hm_raw_df] if not df.empty]
        if not all_tasks_list:
            print("No valid kinematic dataframes to concatenate.")
        else:
            all_tasks_df = pd.concat(all_tasks_list, ignore_index=True)

            if not all_tasks_df.empty:
                metadata_cols_in_all_tasks = ['Patient ID', 'Date of Visit', 'Medication Condition',
                                              'Hand_Performed', 'Task_Type', 'Original_Task_Name_From_File']
                base_kinematic_vars = [col for col in all_tasks_df.columns if col not in metadata_cols_in_all_tasks]

                df_ft_processed = all_tasks_df[all_tasks_df['Task_Type'] == 'ft'].copy()
                if not df_ft_processed.empty:
                    rename_ft_dict = {col: f'ft_{col}' for col in base_kinematic_vars}
                    df_ft_processed.rename(columns=rename_ft_dict, inplace=True)
                    df_ft_processed.drop(columns=['Task_Type'], inplace=True, errors='ignore')

                df_hm_processed = all_tasks_df[all_tasks_df['Task_Type'] == 'hm'].copy()
                if not df_hm_processed.empty:
                    rename_hm_dict = {col: f'hm_{col}' for col in base_kinematic_vars}
                    df_hm_processed.rename(columns=rename_hm_dict, inplace=True)
                    df_hm_processed.drop(columns=['Task_Type'], inplace=True, errors='ignore')

                hand_centric_merge_keys = ['Patient ID', 'Date of Visit', 'Medication Condition', 'Hand_Performed']
                
                if not df_ft_processed.empty: df_ft_processed = df_ft_processed[df_ft_processed['Hand_Performed'].fillna('').astype(str).str.strip() != '']
                if not df_hm_processed.empty: df_hm_processed = df_hm_processed[df_hm_processed['Hand_Performed'].fillna('').astype(str).str.strip() != '']

                temp_dedup_keys = hand_centric_merge_keys + ['Original_Task_Name_From_File']
                if not df_ft_processed.empty:
                    ft_dedup_keys = [k for k in temp_dedup_keys if k in df_ft_processed.columns]
                    df_ft_processed.drop_duplicates(subset=ft_dedup_keys, keep='first', inplace=True)
                if not df_hm_processed.empty:
                    hm_dedup_keys = [k for k in temp_dedup_keys if k in df_hm_processed.columns]
                    df_hm_processed.drop_duplicates(subset=hm_dedup_keys, keep='first', inplace=True)

                if not df_ft_processed.empty: df_ft_processed.drop(columns=['Original_Task_Name_From_File'], errors='ignore', inplace=True)
                if not df_hm_processed.empty: df_hm_processed.drop(columns=['Original_Task_Name_From_File'], errors='ignore', inplace=True)


                if df_ft_processed.empty and df_hm_processed.empty: kinematic_summary_df = pd.DataFrame()
                elif df_ft_processed.empty: kinematic_summary_df = df_hm_processed
                elif df_hm_processed.empty: kinematic_summary_df = df_ft_processed
                else:
                    kinematic_summary_df = pd.merge(
                        df_ft_processed,
                        df_hm_processed,
                        on=hand_centric_merge_keys, how='outer'
                    )

    if not kinematic_summary_df.empty:
        kinematic_summary_df['Patient ID'] = kinematic_summary_df['Patient ID'].astype(str).str.strip()
        try: 
            numeric_pid_kin_sum = pd.to_numeric(kinematic_summary_df['Patient ID'], errors='coerce')
            kinematic_summary_df['Patient ID'] = np.where(numeric_pid_kin_sum.notna(), numeric_pid_kin_sum.astype(int).astype(str), kinematic_summary_df['Patient ID'])
        except Exception as e_final_pid_ks:
             print(f"Warning during final PID standardization in kinematic_summary_df: {e_final_pid_ks}")

        kinematic_summary_df['Date of Visit DT'] = pd.to_datetime(kinematic_summary_df['Date of Visit'], format='%d.%m.%Y', errors='coerce')
        kinematic_summary_df.dropna(subset=['Date of Visit DT', 'Hand_Performed'], how='any', inplace=True)
        
        key_cols_dedup_final = ['Patient ID', 'Date of Visit DT', 'Medication Condition', 'Hand_Performed']
        kinematic_summary_df.drop_duplicates(subset=key_cols_dedup_final, keep='first', inplace=True)
        kinematic_summary_df.sort_values(by=['Patient ID', 'Date of Visit DT'], inplace=True)
        
        kinematic_summary_df['Days Since First Visit'] = kinematic_summary_df.groupby('Patient ID')['Date of Visit DT'].transform(
             lambda x: (x - x.min()).dt.days if pd.notna(x.min()) else np.nan
        )

        # --- MODIFIED SECTION TO INCLUDE AGE ---
        # Load both DaTscan and Age data
        dat_scan_df_loaded = load_dat_scan(input_folder_path)
        age_df_loaded = load_age_data(input_folder_path) # Call the new function
        
        # Prepare a comprehensive patient dataframe by merging DaTscan and Age
        patient_info_df = pd.DataFrame()
        if dat_scan_df_loaded is not None and not dat_scan_df_loaded.empty:
            patient_info_df = dat_scan_df_loaded.copy()
            # If age data is also available, merge it in
            if age_df_loaded is not None and not age_df_loaded.empty:
                print("\nMerging Age data into DaTscan data...")
                # Use a left merge to keep all patients from the main DaTscan file
                patient_info_df = pd.merge(patient_info_df, age_df_loaded, on="Patient ID", how="left")
                print("Merge complete. Patient info DF now contains Age.")
        elif age_df_loaded is not None and not age_df_loaded.empty:
            # Handle case where only Age.csv is present but not DatScan.csv
            patient_info_df = age_df_loaded.copy()

        # Define output path
        output_base_name = f"final_merged_data.csv"
        final_output_path = os.path.join(output_data_dir, output_base_name)

        final_df_to_save = pd.DataFrame()

        # Perform the final merge with the comprehensive patient_info_df
        if not patient_info_df.empty:
            merged_df_with_datscan = merge_dat_scan_hand_specific(kinematic_summary_df, patient_info_df)
            final_df_to_save = merged_df_with_datscan
        else:
            print("\nNo patient-level data (DaTscan or Age) could be loaded. Kinematic summary only.")
            final_df_to_save = kinematic_summary_df
        
        if not final_df_to_save.empty:
            cols_to_exclude = [
                'Contralateral_Caudate_new', 'Contralateral_Putamen_new', 'Contralateral_Striatum_new',
                'Ipsilateral_Caudate_new', 'Ipsilateral_Putamen_new', 'Ipsilateral_Striatum_new', 
                'ft_Kinematic_Task', 'hm_Kinematic_Task',
                'Original_Task_Name_From_File', 
                'Date of Visit DT'
            ]
            cols_with_suffix_to_exclude = [col for col in final_df_to_save.columns if '_InfoSuffix' in col]
            cols_to_exclude.extend(cols_with_suffix_to_exclude)
            
            # Add 'Age' and 'Sex' to the ordered columns for final output
            ordered_cols = ['Patient ID', 'Date of Visit', 'Medication Condition', 'Hand_Performed', 'Days Since First Visit', 'Age', 'Sex']
            
            ft_data_cols = sorted([c for c in final_df_to_save.columns if c.startswith('ft_') and c not in ordered_cols and c not in cols_to_exclude])
            ordered_cols.extend(ft_data_cols)
            hm_data_cols = sorted([c for c in final_df_to_save.columns if c.startswith('hm_') and c not in ordered_cols and c not in cols_to_exclude])
            ordered_cols.extend(hm_data_cols)
            
            contralateral_cols_final = sorted([c for c in final_df_to_save.columns if c.startswith('Contralateral_') and '_new' not in c and c not in ordered_cols and c not in cols_to_exclude])
            ordered_cols.extend(contralateral_cols_final)
            ipsilateral_cols_final = sorted([c for c in final_df_to_save.columns if c.startswith('Ipsilateral_') and '_new' not in c and c not in ordered_cols and c not in cols_to_exclude])
            ordered_cols.extend(ipsilateral_cols_final)
            
            if 'Date of Scan' in final_df_to_save.columns and 'Date of Scan' not in ordered_cols and 'Date of Scan' not in cols_to_exclude:
                ordered_cols.append('Date of Scan')
            
            other_datscan_cols = []
            if dat_scan_df_loaded is not None:
                other_datscan_cols = [
                    c for c in dat_scan_df_loaded.columns 
                    if c in final_df_to_save.columns and 
                       c not in ordered_cols and 
                       c not in cols_to_exclude and
                       not c.endswith("_Z") and not c.endswith("_Raw")
                ]
                if "Patient ID" in other_datscan_cols: other_datscan_cols.remove("Patient ID")
                if "Date of Scan" in other_datscan_cols: other_datscan_cols.remove("Date of Scan")
                ordered_cols.extend(sorted(other_datscan_cols))

            final_selected_cols = [c for c in ordered_cols if c in final_df_to_save.columns and c not in cols_to_exclude]
            remaining_existing_cols = sorted([c for c in final_df_to_save.columns if c not in final_selected_cols and c not in cols_to_exclude])
            final_selected_cols.extend(remaining_existing_cols)
            
            final_selected_cols = list(pd.Series(final_selected_cols).drop_duplicates().tolist())

            final_df_to_save = final_df_to_save[final_selected_cols].copy()

            try:
                final_df_to_save.to_csv(final_output_path, index=False, sep=';', decimal='.')
                print(f"\nFinal hand-centric merged summary (with ipsi/contra Raw & Z & Age) saved to: {final_output_path}")
                print(f"Final DataFrame shape: {final_df_to_save.shape}")
                print(f"Final columns: {final_df_to_save.columns.tolist()}")
            except Exception as e_save: 
                print(f"Failed to save final summary CSV: {e_save}")
                print(f"Attempted columns for saving: {final_df_to_save.columns.tolist()}")
        else:
            print("\nFinal DataFrame to save is empty. No output file created.")
            
    else:
        print("\nKinematic summary DataFrame is empty after initial processing. Cannot create output files.")

print("\n--- Preprocessing script execution finished ---")