# --- START OF FILE Run Summarize CSVs.py (Updated) ---

import os
import re
import pandas as pd
import csv
import numpy as np  # Added to help check for non-finite values

# Get the current script directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder containing all CSV files
input_folder_path = os.path.join(script_dir, 'Input')

def detect_separator(file_path):
    """Detects the CSV separator."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file: # Added encoding safety
            sample = file.read(2048)  # Read a larger sample
            sniffer = csv.Sniffer()
            # Provide possible delimiters if sniff fails often
            try:
                dialect = sniffer.sniff(sample, delimiters=',;')
                return dialect.delimiter
            except csv.Error:
                # Fallback or make an educated guess if sniff fails
                if ';' in sample and ',' not in sample:
                     return ';'
                return ',' # Default to comma
    except Exception as e:
        print(f"Warning: Could not detect separator for {os.path.basename(file_path)} - {e}. Assuming ','.")
        return ','


def parse_kinematic_filename(filename):
    """
    Expected format:
      DateOfVisit_PatientID_MedCondition_KinematicTask - HandCondition.csv
      Handles DateOfVisit as YYYY-MM-DD or YYYYMMDD.
    Example:
      2016-10-17_078_Off_Hand movement - Left.csv
      20190919_149_Off_Finger Tap - Left_thumbsize.csv
    """
    base_name = os.path.splitext(filename)[0]
    # Handle potential extra suffixes like _thumbsize
    base_name = base_name.replace('_thumbsize', '')
    parts = base_name.split('_')

    if len(parts) < 4:
        print(f"Skipping file {filename}: unexpected filename format (too few underscores)")
        return None

    date_of_visit = parts[0]
    patient_id = parts[1]
    med_condition = parts[2]

    # Combine remaining parts for task/hand processing
    task_hand_part = "_".join(parts[3:])

    # The task part is expected to contain the kinematic task and hand condition,
    # separated by " - "
    kinematic_task = ""
    hand_condition = ""
    if ' - ' in task_hand_part:
        task_parts = task_hand_part.split(' - ')
        if len(task_parts) == 2:
            kinematic_task = task_parts[0].strip()
            hand_condition = task_parts[1].strip()
        else:
            # Fallback if multiple ' - ' exist, take first part as task
            kinematic_task = task_parts[0].strip()
            print(f"Warning: Multiple ' - ' found in task part for {filename}. Using '{kinematic_task}'.")
    else:
        # If no ' - ', assume the whole part is the task (might indicate missing hand condition)
        kinematic_task = task_hand_part.strip()
        print(f"Warning: Could not find ' - ' separator for hand condition in {filename}. Hand condition might be missing.")

    # --- FIX 1: Flexible Date Parsing ---
    try:
        # Try YYYY-MM-DD first
        date_of_visit_dt = pd.to_datetime(date_of_visit, format='%Y-%m-%d')
    except ValueError:
        try:
            # Try YYYYMMDD if the first format failed
            date_of_visit_dt = pd.to_datetime(date_of_visit, format='%Y%m%d')
        except Exception as e:
            # Use f-string for better error message
            print(f"Skipping file {filename}: could not parse date '{date_of_visit}' with known formats YYYY-MM-DD or YYYYMMDD.")
            return None # Explicitly return None if date parsing fails

    date_of_visit_str = date_of_visit_dt.strftime('%d.%m.%Y')
    # --- End Fix 1 ---

    # --- FIX 3: Standardize Medication Condition ---
    med_condition_standardized = med_condition.lower()
    # --- End Fix 3 ---

    # Return standardized values
    return date_of_visit_str, patient_id, med_condition_standardized, kinematic_task, hand_condition

def process_kinematic_subfolder(folder_path, prefix):
    """
    Process all CSV files in a specific subfolder.
    Handles CSVs with or without a header 'Attribute,Value'.
    Renames task-specific metadata and all kinematic variable columns with the given prefix.
    Standardizes Medication Condition to lowercase.
    """
    summary_list = []

    print(f"\nProcessing subfolder: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Subfolder not found: {folder_path}")
        return pd.DataFrame()

    processed_files = 0
    skipped_files = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, filename)
        parsed = parse_kinematic_filename(filename)
        if parsed is None:
            skipped_files += 1
            continue

        date_of_visit_str, patient_id, med_condition, kinematic_task, hand_condition = parsed
        file_separator = detect_separator(file_path)

        try:
            # --- Dynamically Handle Header ---
            header_to_use = 0 # Default to assuming header exists (like for Hand_Movements)
            names_to_use = None
            try:
                # Peek at the first line to check for header
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                # Define expected header based on detected separator
                expected_header_str = f"Attribute{file_separator}Value"
                # Check if first line looks like the header (case-insensitive, ignore spaces)
                if first_line.lower().replace(' ', '') != expected_header_str.lower():
                    # If first line doesn't match, assume NO header
                    print(f"Note: Assuming NO header for {filename} based on first line: '{first_line}'")
                    header_to_use = None
                    names_to_use = ['Attribute', 'Value'] # Assign column names manually
                else:
                     print(f"Note: Assuming header exists for {filename}.")

            except Exception as e:
                 print(f"Warning: Could not read first line of {filename} to check header: {e}. Proceeding with default assumption (header=0).")

            # Read CSV using determined header setting
            data_df = pd.read_csv(file_path, sep=file_separator, header=header_to_use, names=names_to_use, encoding='utf-8')
            # --- End Dynamic Header Handling ---

            # --- Processing Logic (remains mostly the same) ---
            # Check if columns are correct now (should always be Attribute, Value)
            if 'Attribute' not in data_df.columns or 'Value' not in data_df.columns:
                 print(f"Skipping file {filename}: Columns 'Attribute', 'Value' not found after read attempt. Columns found: {data_df.columns.tolist()}")
                 skipped_files += 1
                 continue

            # Lowercase the *content* of the 'Attribute' column
            data_df['Attribute'] = data_df['Attribute'].str.lower()

            # Convert 'Value' column to numeric
            data_df['Value'] = pd.to_numeric(data_df['Value'].astype(str).str.strip().str.replace(',', '.'), errors='coerce')

            # Drop rows where 'Value' became NaN
            initial_rows = len(data_df)
            data_df.dropna(subset=['Value'], inplace=True)
            if len(data_df) < initial_rows:
                 print(f"Note: Dropped {initial_rows - len(data_df)} rows with non-numeric 'Value' from {filename}")

            if data_df.empty:
                 print(f"Skipping file {filename}: No valid numeric data found after cleaning 'Value' column.")
                 skipped_files += 1
                 continue

            # Handle duplicates
            if data_df['Attribute'].duplicated().any():
                print(f"Warning: Duplicate attributes found in {filename}. Keeping first occurrence.")
                data_df = data_df.drop_duplicates(subset=['Attribute'], keep='first')

            variables = data_df.set_index('Attribute')['Value'].to_dict()
            variables = {f"{prefix}_{k}": v for k, v in variables.items()}

        except Exception as e:
            print(f"Skipping file {filename}: could not read or process CSV file. Error: {repr(e)}")
            skipped_files += 1
            continue

        # Create summary row
        summary_row = {
            'Patient ID': patient_id,
            'Date of Visit': date_of_visit_str,
            'Medication Condition': med_condition,
            f"{prefix}_Kinematic Task": kinematic_task,
            f"{prefix}_Hand Condition": hand_condition
        }
        summary_row.update(variables)
        summary_list.append(summary_row)
        processed_files += 1

    print(f"Finished processing {folder_path}: Processed {processed_files} files, Skipped {skipped_files} files.")
    return pd.DataFrame(summary_list)


def load_dat_scan(input_folder):
    """
    Load and preprocess the DatScan.csv file.
    Renames columns, converts the date column, drops the Gender column,
    and handles non-finite Patient IDs.
    """
    dat_scan_path = os.path.join(input_folder, 'DatScan.csv')
    try:
        # Try ';' first, fallback to ',' - REMOVED 'errors' argument
        try:
             dat_scan_df = pd.read_csv(dat_scan_path, delimiter=';', encoding='utf-8') # Removed errors='ignore'
             print("Read DatScan.csv with ';' delimiter.")
        except Exception:
             dat_scan_df = pd.read_csv(dat_scan_path, delimiter=',', encoding='utf-8') # Removed errors='ignore'
             print("Read DatScan.csv with ',' delimiter.")
    except Exception as e:
        # Use repr(e) for potentially more detail
        raise Exception(f"Error reading DatScan.csv: {repr(e)}") # Use repr(e)

    # --- Renaming and processing logic (remains the same as the previous version) ---
    current_cols_map = {col.strip().lower(): col for col in dat_scan_df.columns}
    rename_dict_flexible = {
        "Patient ID": ["no.", "patient id"],
        "Date of Scan": ["date of scan (dos)", "date of scan"],
        "Gender": ["gender       0 = female\n1= male", "gender"],
        "Striatum_Right_Z": ["striatum right: z-werte (new software)"],
        "Striatum_Left_Z": ["striatum left: z-werte (new software)"],
        "Putamen_Right_Z": ["putamen right: z-werte (new software)"],
        "Putamen_Left_Z": ["putamen left: z-werte (new software)"],
        "Caudate_Right_Z": ["caudate right: z-werte (new software)"],
        "Caudate_Left_Z": ["caudate left: z-werte (new software)"],
        "Mean_striatum_Z": ["mean striatum: z-werte (new software)"],
        "Mean_Putamen_Z": ["mean putamen: z-werte (new software)"],
        "Mean_Caudate_Z": ["mean caudate: z-werte (new software)"],
        "Striatum_Right_new": ["striatum right: new software"],
        "Striatum_Left_new": ["striatum leftt: new software", "striatum left: new software"],
        "Putamen_Right_new": ["putamen rightt: new software", "putamen right: new software"],
        "Putamen_Left_new": ["putamen leftt: new software", "putamen left: new software"],
        "Caudate_Right_new": ["caudate rightt: new software", "caudate right: new software"],
        "Caudate_Left_new": ["caudate leftt: new software", "caudate left: new software"],
        "Mean_striatum_new": ["mean striatumt: new software", "mean striatum: new software"],
        "Mean_Putamen_new": ["mean putament: new software", "mean putamen: new software"],
        "Mean_Caudate_new": ["mean caudatet: new software", "mean caudate: new software"]
    }
    actual_rename_dict = {}
    found_targets = set()
    missing_sources = []
    for target_name, potential_sources in rename_dict_flexible.items():
        found = False
        for source in potential_sources:
            if source in current_cols_map:
                actual_rename_dict[current_cols_map[source]] = target_name
                found_targets.add(target_name)
                found = True
                break
        if not found:
            missing_sources.append(f"'{target_name}' (tried: {potential_sources})")
    if missing_sources:
         print("\n--- Warning: Potential Missing DatScan Columns ---")
         print("Could not find source columns for the following target names:")
         for missing in missing_sources: print(f"- {missing}")
         print("Check the exact column names in your DatScan.csv file.")
         print("-------------------------------------------------\n")
    dat_scan_df.rename(columns=actual_rename_dict, inplace=True)
    print(f"Renamed DatScan columns. Kept: {list(found_targets)}")
    essential_cols = ["Patient ID", "Date of Scan"]
    final_dat_scan_cols = essential_cols + list(found_targets - set(essential_cols))
    dat_scan_df = dat_scan_df[[col for col in final_dat_scan_cols if col in dat_scan_df.columns]]
    if "Date of Scan" in dat_scan_df.columns:
        dat_scan_df['Date of Scan'] = pd.to_datetime(dat_scan_df['Date of Scan'], errors='coerce', dayfirst=True)
        if dat_scan_df['Date of Scan'].isnull().any(): print("Warning: Some 'Date of Scan' values in DatScan.csv could not be parsed.")
    else: print("Warning: 'Date of Scan' column not found after renaming.")
    if "Patient ID" not in dat_scan_df.columns: raise Exception("Critical Error: 'Patient ID' column not found in DatScan data after processing.")
    dat_scan_df = dat_scan_df.dropna(subset=['Patient ID'])
    dat_scan_df['Patient ID'] = pd.to_numeric(dat_scan_df['Patient ID'], errors='coerce')
    dat_scan_df = dat_scan_df.dropna(subset=['Patient ID'])
    dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(int).astype(str)
    if 'Gender' in dat_scan_df.columns: dat_scan_df.drop(columns=['Gender'], inplace=True)
    value_cols = [col for col in dat_scan_df.columns if '_Z' in col or '_new' in col]
    for col in value_cols:
        if col in dat_scan_df.columns: dat_scan_df[col] = pd.to_numeric(dat_scan_df[col].astype(str).str.replace(',', '.'), errors='coerce')
    print(f"Processed DatScan data. Shape: {dat_scan_df.shape}")
    print("Debug: Loaded DatScan.csv. Patient IDs sample:", dat_scan_df['Patient ID'].unique()[:20])
    return dat_scan_df


def merge_dat_scan(kinematic_df, dat_scan_df):
    """
    Merge the kinematic summary with DatScan imaging data.
    For each row, only the contralateral imaging data (based on Hand Condition) are retained.
    """
    # Ensure 'Patient ID' in kinematic_df is also standardized as string
    if 'Patient ID' not in kinematic_df.columns:
         print("Error: 'Patient ID' missing from kinematic_df. Cannot merge.")
         return kinematic_df # Return original kinematic data

    # Standardize kinematic Patient ID just before merge
    def convert_pid_kinematic(pid):
        try:
            # Convert to numeric first to handle potential '.0', then int, then string
            return str(int(float(pid)))
        except (ValueError, TypeError):
             return str(pid).strip() # Keep as string if conversion fails

    kinematic_df['Patient ID'] = kinematic_df['Patient ID'].apply(convert_pid_kinematic)

    # Prepare DatScan Patient ID just before merge (already done in load_dat_scan, but ensures consistency)
    dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(str)

    # Identify common Patient IDs
    kinematic_pids = set(kinematic_df['Patient ID'].unique())
    datscan_pids = set(dat_scan_df['Patient ID'].unique())
    common_pids = kinematic_pids.intersection(datscan_pids)
    print(f"Found {len(common_pids)} common Patient IDs for merging.")
    if not common_pids:
         print("Warning: No common Patient IDs found between kinematic data and DatScan data. Merge will likely yield no matches.")

    # Merge on common key(s). Merge on Patient ID only for simplicity.
    # Use 'left' merge to keep all kinematic rows.
    print("Merging kinematic and DatScan data on 'Patient ID'...")
    merged_df = pd.merge(kinematic_df, dat_scan_df, on="Patient ID", how="left")
    print(f"Shape after merge: {merged_df.shape}")

    # Define which imaging keys are side-specific. (Base names, without _Right/_Left)
    # Use the target names from the rename dictionary keys
    imaging_keys_base = ['Striatum', 'Putamen', 'Caudate'] # Base names
    imaging_versions = ['Z', 'new'] # Measurement types

    def select_contralateral(row):
        # Determine hand condition. Check ft first, then hm.
        hand = None
        if f"ft_Hand Condition" in row and pd.notna(row[f"ft_Hand Condition"]):
            hand = row[f"ft_Hand Condition"]
        elif f"hm_Hand Condition" in row and pd.notna(row[f"hm_Hand Condition"]):
            hand = row[f"hm_Hand Condition"]
        # Fallback if somehow present without prefix (less likely now)
        elif 'Hand Condition' in row and pd.notna(row['Hand Condition']):
             hand = row['Hand Condition']

        if hand is None:
             # print(f"Debug: No hand condition found for Patient ID {row.get('Patient ID', 'N/A')}, Visit {row.get('Date of Visit', 'N/A')}. Cannot determine contralateral.")
             return row # Cannot determine contralateral side

        hand = str(hand).strip().lower()
        side_to_keep = None # Which side's data to keep (e.g., 'Right' means keep Striatum_Right_Z)
        if hand == 'left':
            side_to_keep = 'Right'
        elif hand == 'right':
            side_to_keep = 'Left'
        else:
            # print(f"Debug: Unrecognized hand condition '{hand}' for Patient ID {row.get('Patient ID', 'N/A')}. Cannot determine contralateral.")
            return row # Unrecognized hand condition

        # For each imaging key and for each relevant version, keep only the contralateral side data.
        for key_base in imaging_keys_base:
            for version in imaging_versions:
                # Construct the column name for the side we want to KEEP
                source_col_name = f"{key_base}_{side_to_keep}_{version}"
                # Construct the new generic 'Contralateral' column name
                new_col_name = f"Contralateral_{key_base}_{version}"

                # Assign the value from the source column to the new column
                # Use .get() to handle cases where the source column might be missing (e.g., DatScan merge failed for this row)
                row[new_col_name] = row.get(source_col_name, np.nan) # Use NaN if source missing

        return row

    print("Applying contralateral selection logic...")
    merged_df = merged_df.apply(select_contralateral, axis=1)

    # Identify all original side-specific DatScan columns to drop
    cols_to_drop = []
    for key_base in imaging_keys_base:
        for version in imaging_versions:
            for side in ['Left', 'Right']:
                 col_name = f"{key_base}_{side}_{version}"
                 if col_name in merged_df.columns:
                      cols_to_drop.append(col_name)
            # Also drop mean columns if they exist (not needed after contralateral selection)
            mean_col_name = f"Mean_{key_base.lower()}_{version}" # e.g., Mean_striatum_Z
            if mean_col_name in merged_df.columns:
                 cols_to_drop.append(mean_col_name)


    print(f"Dropping original side-specific and mean DatScan columns: {cols_to_drop}")
    # Check existence before dropping to avoid errors
    cols_to_drop_existing = [col for col in cols_to_drop if col in merged_df.columns]
    merged_df.drop(columns=cols_to_drop_existing, inplace=True, errors='ignore')
    print(f"Shape after dropping original DatScan columns: {merged_df.shape}")

    return merged_df

# ========================
# Main Execution Block
# ========================
if __name__ == '__main__':
    # Define paths for the two kinematic subfolders
    fingertapping_path = os.path.join(input_folder_path, 'Fingertapping')
    hand_movements_path = os.path.join(input_folder_path, 'Hand_Movements')

    # Process each subfolder with its respective prefix.
    ft_df = process_kinematic_subfolder(fingertapping_path, "ft")

    # --- Debug Print ft_df ---
    print("\n--- Finger Tapping DataFrame (ft_df) ---")
    print(f"ft_df is empty: {ft_df.empty}")
    if not ft_df.empty:
        print("Shape:", ft_df.shape)
        print("First 5 rows:")
        print(ft_df.head())
        print("\nInfo (columns, types, non-null counts):")
        ft_df.info()
        print("\nMerge Keys (first 5):")
        print(ft_df[["Patient ID", "Date of Visit", "Medication Condition"]].head())
    print("----------------------------------------\n")

    hm_df = process_kinematic_subfolder(hand_movements_path, "hm")

    # --- Debug Print hm_df ---
    print("\n--- Hand Movement DataFrame (hm_df) ---")
    print(f"hm_df is empty: {hm_df.empty}")
    if not hm_df.empty:
        print("Shape:", hm_df.shape)
        print("First 5 rows:")
        print(hm_df.head())
        print("\nInfo (columns, types, non-null counts):")
        hm_df.info()
        print("\nMerge Keys (first 5):")
        print(hm_df[["Patient ID", "Date of Visit", "Medication Condition"]].head())
    print("---------------------------------------\n")

    # Check if both DataFrames are non-empty before merging
    if ft_df.empty and hm_df.empty:
        print("Both ft_df and hm_df are empty. Cannot proceed.")
        kinematic_summary_df = pd.DataFrame() # Assign empty df
    elif ft_df.empty:
        print("ft_df is empty. Using only hm_df.")
        kinematic_summary_df = hm_df
    elif hm_df.empty:
        print("hm_df is empty. Using only ft_df.")
        kinematic_summary_df = ft_df
    else:
        # Merge the two kinematic DataFrames on common keys
        print("Attempting merge between ft_df and hm_df...")
        kinematic_summary_df = pd.merge(ft_df, hm_df, on=["Patient ID", "Date of Visit", "Medication Condition"], how="outer")
        print(f"Merge successful. Shape of kinematic_summary_df: {kinematic_summary_df.shape}")

    # Proceed only if kinematic_summary_df is not empty
    if not kinematic_summary_df.empty:
        # --- Debug Print merged info ---
        print("\n--- Merged Kinematic DataFrame Info (kinematic_summary_df) ---")
        print("Shape:", kinematic_summary_df.shape)
        print("\nInfo (columns, types, non-null counts):")
        kinematic_summary_df.info()
        # Check if hm columns exist at all
        hm_cols_exist = [col for col in kinematic_summary_df.columns if 'hm_' in col and 'Hand Condition' not in col and 'Kinematic Task' not in col]
        ft_cols_exist = [col for col in kinematic_summary_df.columns if 'ft_' in col and 'Hand Condition' not in col and 'Kinematic Task' not in col]
        print(f"\nNumber of 'ft_' kinematic columns found: {len(ft_cols_exist)}")
        print(f"Number of 'hm_' kinematic columns found: {len(hm_cols_exist)}")
        print("\nExample rows (showing potential merge outcomes - NaN indicates missing data for that task/visit):")
        # Show rows where one task is present but the other might be missing or rows that merged successfully
        print(kinematic_summary_df.head(10))
        print("----------------------------------------------------\n")

        # Standardize Patient ID further if needed (already done partially)
        def convert_pid_final(pid):
            try:
                return str(int(float(pid))) # Handles ints, floats, and strings representing them
            except (ValueError, TypeError):
                return str(pid).strip() # Keep as string otherwise
        if 'Patient ID' in kinematic_summary_df.columns:
             kinematic_summary_df['Patient ID'] = kinematic_summary_df['Patient ID'].apply(convert_pid_final)


        # Debug: print kinematic Patient IDs after merge and standardization
        print("Kinematic Patient IDs in summary:", kinematic_summary_df['Patient ID'].unique())

        # Convert 'Date of Visit' back to datetime for sorting and calculation
        # Coerce errors to handle any potential invalid date strings remaining
        kinematic_summary_df['Date of Visit DT'] = pd.to_datetime(kinematic_summary_df['Date of Visit'], format='%d.%m.%Y', errors='coerce')
        kinematic_summary_df.dropna(subset=['Date of Visit DT'], inplace=True) # Drop rows with invalid dates

        # Remove potential duplicate rows based on key identifiers
        key_cols = ["Patient ID", "Date of Visit", "Medication Condition"]
        # Also consider hand conditions if they should define uniqueness
        if "ft_Hand Condition" in kinematic_summary_df.columns: key_cols.append("ft_Hand Condition")
        if "hm_Hand Condition" in kinematic_summary_df.columns: key_cols.append("hm_Hand Condition")
        initial_rows = len(kinematic_summary_df)
        kinematic_summary_df.drop_duplicates(subset=key_cols, keep='first', inplace=True)
        print(f"Dropped {initial_rows - len(kinematic_summary_df)} duplicate rows based on keys: {key_cols}")


        # Calculate 'Days Since First Visit' for each patient
        # Sort by patient and date first to ensure correct calculation
        kinematic_summary_df.sort_values(by=['Patient ID', 'Date of Visit DT'], inplace=True)
        kinematic_summary_df['Days Since First Visit'] = kinematic_summary_df.groupby('Patient ID')['Date of Visit DT'].transform(
            lambda x: (x - x.min()).dt.days)

        # Rearrange columns to position 'Days Since First Visit' after 'Medication Condition'
        cols = list(kinematic_summary_df.columns)
        if 'Days Since First Visit' in cols:
            cols.remove('Days Since First Visit')
        if 'Medication Condition' in cols:
            insert_at = cols.index('Medication Condition') + 1
        else:
            insert_at = 3 # Fallback position
        cols.insert(insert_at, 'Days Since First Visit')

        # Remove the temporary datetime column
        if 'Date of Visit DT' in cols:
            cols.remove('Date of Visit DT')
        kinematic_summary_df = kinematic_summary_df[cols]

        # Convert 'Date of Visit' back to string format DD.MM.YYYY for saving (it remained object type)
        # No conversion needed if it's already the correct string format

        # Save the kinematic summary DataFrame to a CSV file
        summary_file_path = os.path.join(input_folder_path, 'summary.csv')
        try:
            # Use semicolon separator for better Excel compatibility with European settings
            kinematic_summary_df.to_csv(summary_file_path, index=False, sep=';', decimal='.')
            print(f"\nKinematic summary CSV file created at: {summary_file_path}")
        except Exception as e:
            print(f"Failed to save kinematic summary CSV at {summary_file_path}: {e}")

        # Load DatScan data.
        try:
            dat_scan_df = load_dat_scan(input_folder_path)
        except Exception as e:
            print(f"Error loading DatScan data: {e}")
            dat_scan_df = None

        if dat_scan_df is not None and not dat_scan_df.empty:
            # Merge kinematic summary with DatScan data (contralateral imaging columns).
            merged_df = merge_dat_scan(kinematic_summary_df, dat_scan_df)

            # Save the merged summary DataFrame to a new CSV file in the same folder.
            merged_summary_file_path = os.path.join(input_folder_path, 'merged_summary.csv')
            try:
                # Use semicolon separator for better Excel compatibility
                merged_df.to_csv(merged_summary_file_path, index=False, sep=';', decimal='.')
                print(f"Merged summary CSV file created at: {merged_summary_file_path}")
            except Exception as e:
                print(f"Failed to save merged summary CSV at {merged_summary_file_path}: {e}")
        elif dat_scan_df is None:
             print("DatScan data could not be loaded. Skipping merge.")
        else: # dat_scan_df is empty
             print("DatScan data was loaded but is empty. Skipping merge.")

    else: # kinematic_summary_df is empty
        print("Kinematic summary DataFrame is empty after processing subfolders. Cannot create summary or merged files.")

print("\nScript execution finished.")
# --- END OF FILE ---