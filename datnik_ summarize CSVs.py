# --- START OF FILE datnik_summarize_CSVs.py (Cleaned) ---

import os
import re
import pandas as pd
import csv
import numpy as np

# Get the current script directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder containing all CSV files (Assumes Input is sibling to script)
input_folder_path = os.path.join(script_dir, 'Input')

def detect_separator(file_path):
    """Detects the CSV separator."""
    try:
        # Increase sample size for better detection
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file: # Use replace for encoding errors
            sample = file.read(4096)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=',;')
                return dialect.delimiter
            except csv.Error:
                # Fallback logic if sniffing fails
                semicolon_count = sample.count(';')
                comma_count = sample.count(',')
                # Basic heuristic: prefer semicolon if clearly more frequent or only one present
                if semicolon_count > comma_count and comma_count == 0:
                    return ';'
                elif comma_count > semicolon_count and semicolon_count == 0:
                    return ','
                elif semicolon_count > 0: # Default to semicolon if both present? Adjust if needed.
                    return ';'
                else:
                    return ',' # Default fallback
    except Exception as e:
        print(f"Warning: Could not detect separator for {os.path.basename(file_path)} - {e}. Assuming ','.")
        return ','


def parse_kinematic_filename(filename):
    """
    Expected format:
      DateOfVisit_PatientID_MedCondition_KinematicTask - HandCondition.csv
      Handles DateOfVisit as YYYY-MM-DD or YYYYMMDD.
      MedCondition expected to be 'Off' or 'On' (case-insensitive).
      Cleans leading/trailing spaces from MedCondition.
    """
    base_name = os.path.splitext(filename)[0]
    # Handle potential extra suffixes like _thumbsize BEFORE splitting
    base_name = base_name.replace('_thumbsize', '')
    parts = base_name.split('_')

    if len(parts) < 4:
        print(f"Skipping file {filename}: unexpected filename format (too few underscores: {len(parts)-1})")
        return None

    date_of_visit_raw = parts[0]
    patient_id_raw = parts[1]
    med_condition_raw = parts[2]

    task_hand_part = "_".join(parts[3:])
    kinematic_task = ""
    hand_condition = ""
    if ' - ' in task_hand_part:
        task_parts = task_hand_part.split(' - ', 1) # Split only once
        kinematic_task = task_parts[0].strip()
        if len(task_parts) > 1:
            hand_condition = task_parts[1].strip()
        else:
             print(f"Warning: Found ' - ' but no text after it for hand condition in {filename}.")
    else:
        kinematic_task = task_hand_part.strip()
        print(f"Warning: Could not find ' - ' separator for hand condition in {filename}.")

    # --- Validate and Parse Date ---
    try:
        # Try YYYY-MM-DD first
        date_of_visit_dt = pd.to_datetime(date_of_visit_raw, format='%Y-%m-%d')
    except ValueError:
        try:
            # Try YYYYMMDD if the first format failed
            date_of_visit_dt = pd.to_datetime(date_of_visit_raw, format='%Y%m%d')
        except ValueError: # More specific error catch
            print(f"Skipping file {filename}: could not parse date '{date_of_visit_raw}' with known formats YYYY-MM-DD or YYYYMMDD.")
            return None
        except Exception as e_date: # Catch other potential date errors
            print(f"Skipping file {filename}: unexpected error parsing date '{date_of_visit_raw}': {e_date}")
            return None

    date_of_visit_str = date_of_visit_dt.strftime('%d.%m.%Y')

    # --- Clean Patient ID ---
    patient_id = str(patient_id_raw).strip()

    # --- Clean and Standardize Medication Condition ---
    med_condition_standardized = med_condition_raw.strip().lower()
    if med_condition_standardized not in ['off', 'on']:
         print(f"Warning: Unexpected Medication Condition '{med_condition_raw}' -> '{med_condition_standardized}' found in {filename}. Using '{med_condition_standardized}'. Expected 'off' or 'on'.")

    return date_of_visit_str, patient_id, med_condition_standardized, kinematic_task, hand_condition


def process_kinematic_subfolder(base_folder_path, prefix):
    """
    Processes all CSV files in a specific base task folder AND its 'MedON' subfolder.
    Handles CSVs with or without a header 'Attribute,Value'.
    Renames task-specific metadata and all kinematic variable columns with the given prefix.
    Uses cleaned Medication Condition ('off' or 'on').
    """
    summary_list = []
    medon_subfolder_path = os.path.join(base_folder_path, 'MedON')
    folders_to_scan = [base_folder_path]
    if os.path.isdir(medon_subfolder_path):
        folders_to_scan.append(medon_subfolder_path)
        print(f"\nFound 'MedON' subfolder for {prefix}: {medon_subfolder_path}")
    # else: print(f"Note: No 'MedON' subfolder found for {prefix}")

    print(f"Processing task '{prefix.upper()}' in folders: {folders_to_scan}")
    total_processed_files = 0
    total_skipped_files = 0

    for current_folder in folders_to_scan:
        print(f"--- Scanning folder: {current_folder} ---")
        if not os.path.isdir(current_folder):
            print(f"Warning: Folder not found, skipping: {current_folder}")
            continue

        processed_in_folder = 0
        skipped_in_folder = 0

        for filename in os.listdir(current_folder):
            if not filename.endswith('.csv') or filename.startswith('.'): # Skip hidden files
                continue

            file_path = os.path.join(current_folder, filename)

            # --- Parse Filename ---
            parsed = parse_kinematic_filename(filename)
            if parsed is None:
                skipped_in_folder += 1
                continue
            date_of_visit_str, patient_id, med_condition, kinematic_task, hand_condition = parsed

            # --- Detect Separator ---
            file_separator = detect_separator(file_path)

            # --- Read and Process CSV ---
            try:
                # -- Determine header dynamically --
                header_to_use = 0 # Default: assume header exists
                names_to_use = None
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        first_line = f.readline().strip()
                    # Basic check: does it look like "Attribute<sep>Value"?
                    expected_header_pat = re.compile(f"attribute\\s*{re.escape(file_separator)}\\s*value", re.IGNORECASE)
                    if not expected_header_pat.match(first_line):
                        # If not, assume NO header and provide names
                        header_to_use = None
                        names_to_use = ['Attribute', 'Value']
                        # print(f"Note: Assuming NO header for {filename}") # Optional verbose
                except Exception as e_head:
                     print(f"Warning: Could not read first line of {filename} to check header: {e_head}. Proceeding with default assumption (header=0).")

                # -- Read CSV (NO errors='ignore' argument) --
                data_df = pd.read_csv(file_path, sep=file_separator, header=header_to_use, names=names_to_use, encoding='utf-8', on_bad_lines='warn') # Use on_bad_lines
                # Using on_bad_lines='warn' will show warnings for problematic lines but try to continue. Use 'skip' to ignore them silently, or remove for default ('error').


                # -- Validate Columns --
                if 'Attribute' not in data_df.columns or 'Value' not in data_df.columns:
                     print(f"Skipping file {filename}: Columns 'Attribute', 'Value' not found after read. Found: {data_df.columns.tolist()}")
                     skipped_in_folder += 1
                     continue

                # -- Clean and Convert Data --
                data_df['Attribute'] = data_df['Attribute'].astype(str).str.strip().str.lower()
                data_df['Value'] = pd.to_numeric(data_df['Value'].astype(str).str.strip().str.replace(',', '.'), errors='coerce')

                # -- Handle NaNs and Duplicates --
                initial_rows = len(data_df)
                data_df.dropna(subset=['Value'], inplace=True)
                if len(data_df) < initial_rows:
                    print(f"Note: Dropped {initial_rows - len(data_df)} rows with non-numeric 'Value' from {filename}")

                if data_df.empty:
                     # print(f"Skipping file {filename}: No valid numeric data after cleaning.") # Less verbose
                     skipped_in_folder += 1
                     continue

                if data_df['Attribute'].duplicated().any():
                    # print(f"Warning: Duplicate attributes found in {filename}. Keeping first occurrence.") # Less verbose
                    data_df = data_df.drop_duplicates(subset=['Attribute'], keep='first')

                # -- Extract Variables --
                variables = data_df.set_index('Attribute')['Value'].to_dict()
                variables = {f"{prefix}_{k}": v for k, v in variables.items()}

            # --- Catch Errors During File Processing ---
            except pd.errors.EmptyDataError:
                print(f"Skipping file {filename}: File is empty.")
                skipped_in_folder += 1
                continue
            except pd.errors.ParserError as e_parse:
                 print(f"Skipping file {filename}: Failed to parse CSV content. Error: {e_parse}")
                 skipped_in_folder += 1
                 continue
            except Exception as e_proc: # Catch other processing errors
                print(f"Skipping file {filename}: could not read or process CSV. Error: {repr(e_proc)}")
                skipped_in_folder += 1
                continue

            # --- Create Summary Row ---
            summary_row = {
                'Patient ID': patient_id,
                'Date of Visit': date_of_visit_str,
                'Medication Condition': med_condition, # Already cleaned 'off' or 'on'
                f"{prefix}_Kinematic Task": kinematic_task,
                f"{prefix}_Hand Condition": hand_condition
            }
            summary_row.update(variables)
            summary_list.append(summary_row)
            processed_in_folder += 1

        print(f"--- Finished folder scan: Processed: {processed_in_folder}, Skipped: {skipped_in_folder} ---")
        total_processed_files += processed_in_folder
        total_skipped_files += skipped_in_folder

    print(f"\nFinished processing task '{prefix.upper()}': Total Processed {total_processed_files} files, Total Skipped {total_skipped_files} files.")
    return pd.DataFrame(summary_list)


def load_dat_scan(input_folder):
    """
    Load and preprocess the DatScan.csv file.
    """
    dat_scan_path = os.path.join(input_folder, 'DatScan.csv')
    dat_scan_df = None
    print(f"\nAttempting to load DatScan data from: {dat_scan_path}")
    try:
        try:
             # Read with semicolon (NO errors='ignore')
             dat_scan_df = pd.read_csv(dat_scan_path, delimiter=';', encoding='utf-8')
             print("Read DatScan.csv with ';' delimiter.")
        except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError):
             print("Failed read/parse with ';'. Trying ',' delimiter...")
             # Read with comma (NO errors='ignore')
             dat_scan_df = pd.read_csv(dat_scan_path, delimiter=',', encoding='utf-8')
             print("Read DatScan.csv with ',' delimiter.")
        except Exception as e_first: # Catch other errors on first try
             print(f"Error reading DatScan.csv with ';': {repr(e_first)}. Trying ','...")
             try:
                  dat_scan_df = pd.read_csv(dat_scan_path, delimiter=',', encoding='utf-8')
                  print("Read DatScan.csv with ',' delimiter.")
             except Exception as e_second:
                 print(f"Error reading DatScan.csv with ',' as well: {repr(e_second)}")
                 return None
    except FileNotFoundError:
        print(f"Error: DatScan.csv not found at {dat_scan_path}")
        return None
    except Exception as e_load: # Catch errors during the try/except logic itself
        print(f"Unexpected error during DatScan loading process: {repr(e_load)}")
        return None

    if dat_scan_df is None:
        print("DatScan DataFrame could not be loaded.")
        return None

    print(f"DatScan loaded successfully. Initial shape: {dat_scan_df.shape}")

    # --- Renaming logic ---
    current_cols_map = {col.strip().lower(): col for col in dat_scan_df.columns}
    rename_dict_flexible = { # Keep your specific columns here
        "Patient ID": ["no.", "patient id"],
        "Date of Scan": ["date of scan (dos)", "date of scan"],
        "Gender": ["gender       0 = female\n1= male", "gender"],
        "Striatum_Right_Z": ["striatum right: z-werte (new software)"],
        "Striatum_Left_Z": ["striatum left: z-werte (new software)"],
        "Putamen_Right_Z": ["putamen right: z-werte (new software)"],
        "Putamen_Left_Z": ["putamen left: z-werte (new software)"],
        "Caudate_Right_Z": ["caudate right: z-werte (new software)"],
        "Caudate_Left_Z": ["caudate left: z-werte (new software)"],
        # Add other mappings as needed...
    }
    actual_rename_dict = {}
    found_targets = set()
    missing_sources = []
    for target_name, potential_sources in rename_dict_flexible.items():
        found = False
        for source in potential_sources:
            source_clean = source.strip().lower()
            if source_clean in current_cols_map:
                actual_rename_dict[current_cols_map[source_clean]] = target_name
                found_targets.add(target_name)
                found = True
                break
        if not found:
            missing_sources.append(f"'{target_name}' (tried: {potential_sources})")

    if missing_sources:
         print("\n--- Warning: Potential Missing DatScan Columns ---")
         print("Could not find source columns for the following target names:")
         for missing in missing_sources: print(f"- {missing}")
         print("-------------------------------------------------\n")

    dat_scan_df.rename(columns=actual_rename_dict, inplace=True)
    print(f"DatScan columns renamed. Attempted to find: {list(rename_dict_flexible.keys())}")

    # --- Select and Clean Essential Columns ---
    essential_cols = ["Patient ID", "Date of Scan"]
    final_dat_scan_cols = essential_cols + [col for col in rename_dict_flexible if col not in essential_cols]
    # Keep only columns that exist AFTER renaming
    final_dat_scan_cols_present = [col for col in final_dat_scan_cols if col in dat_scan_df.columns]
    if not final_dat_scan_cols_present:
         print("Error: No target DatScan columns found after renaming. Check rename_dict_flexible.")
         return None
    dat_scan_df = dat_scan_df[final_dat_scan_cols_present].copy() # Use .copy()
    print(f"Kept DatScan columns: {final_dat_scan_cols_present}")


    # Clean Date of Scan
    if "Date of Scan" in dat_scan_df.columns:
        dat_scan_df['Date of Scan'] = pd.to_datetime(dat_scan_df['Date of Scan'], errors='coerce', dayfirst=True)
        if dat_scan_df['Date of Scan'].isnull().any():
            print("Warning: Some 'Date of Scan' values in DatScan.csv could not be parsed and were set to NaT.")
    else:
        print("Warning: 'Date of Scan' column not found after renaming. Cannot process scan dates.")

    # Clean Patient ID
    if "Patient ID" not in dat_scan_df.columns:
        print("Critical Error: 'Patient ID' column not found in DatScan data after processing.")
        return None
    dat_scan_df.dropna(subset=['Patient ID'], inplace=True)
    try:
        # Robust conversion: to numeric handles floats, then int, then str
        dat_scan_df['Patient ID'] = pd.to_numeric(dat_scan_df['Patient ID'], errors='coerce')
        dat_scan_df.dropna(subset=['Patient ID'], inplace=True)
        dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(int).astype(str)
    except Exception as e_pid:
        print(f"Warning: Could not robustly convert DatScan Patient ID to string: {e_pid}. Using simple string conversion.")
        dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(str).str.strip()


    # Clean Numeric Values (Z-scores, etc.)
    value_cols = [col for col in dat_scan_df.columns if '_Z' in col or '_new' in col]
    for col in value_cols:
        if col in dat_scan_df.columns:
            original_nan_count = dat_scan_df[col].isnull().sum()
            dat_scan_df[col] = pd.to_numeric(dat_scan_df[col].astype(str).str.replace(',', '.'), errors='coerce')
            new_nan_count = dat_scan_df[col].isnull().sum()
            if new_nan_count > original_nan_count:
                print(f"Note: Coerced non-numeric values to NaN in DatScan column '{col}'.")

    # Drop Gender if present
    if 'Gender' in dat_scan_df.columns:
        dat_scan_df.drop(columns=['Gender'], inplace=True, errors='ignore')

    print(f"Processed DatScan data. Final shape: {dat_scan_df.shape}")
    return dat_scan_df


def merge_dat_scan(kinematic_df, dat_scan_df):
    """
    Merge the kinematic summary with DatScan imaging data.
    Contralateral logic applies per row based on Hand Condition.
    """
    if 'Patient ID' not in kinematic_df.columns:
         print("Error: 'Patient ID' missing from kinematic_df. Cannot merge.")
         return kinematic_df
    if dat_scan_df is None or dat_scan_df.empty:
         print("DatScan DataFrame is empty or None. Skipping merge.")
         return kinematic_df # Return kinematic data unmodified

    print("\nStandardizing Patient IDs before merge...")
    def convert_pid_safe(pid):
        try: # Handle potential float strings like '123.0'
            return str(int(float(pid)))
        except (ValueError, TypeError): # Handle non-numeric, None, etc.
            return str(pid).strip()

    kinematic_df['Patient ID'] = kinematic_df['Patient ID'].apply(convert_pid_safe)
    dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].apply(convert_pid_safe)


    kinematic_pids = set(kinematic_df['Patient ID'].unique())
    datscan_pids = set(dat_scan_df['Patient ID'].unique())
    common_pids = kinematic_pids.intersection(datscan_pids)
    print(f"Found {len(common_pids)} common Patient IDs for merging kinematic and DatScan.")
    if not common_pids: print("Warning: No common Patient IDs found!")

    print("Merging kinematic and DatScan data on 'Patient ID'...")
    # Validate merge columns
    if 'Patient ID' not in kinematic_df.columns or 'Patient ID' not in dat_scan_df.columns:
        print("Error: 'Patient ID' column missing in one of the dataframes before merge.")
        return kinematic_df

    merged_df = pd.merge(kinematic_df, dat_scan_df, on="Patient ID", how="left", suffixes=('', '_DatScan')) # Add suffix to avoid duplicate cols
    print(f"Shape after DatScan merge: {merged_df.shape}")

    # Define imaging keys to process for contralateral selection
    # Make sure these BASE names match the columns available AFTER renaming in load_dat_scan
    imaging_keys_base = ['Striatum', 'Putamen', 'Caudate']
    imaging_versions = ['Z', 'new'] # Add 'new' if you have those columns too

    def select_contralateral(row):
        """Applies contralateral selection logic to a row (pandas Series)."""
        hand = None
        # Determine hand condition, checking FT then HM
        ft_hand_col = 'ft_Hand Condition'
        hm_hand_col = 'hm_Hand Condition'
        if ft_hand_col in row.index and pd.notna(row[ft_hand_col]):
            hand = row[ft_hand_col]
        elif hm_hand_col in row.index and pd.notna(row[hm_hand_col]):
            hand = row[hm_hand_col]

        if hand is None: return row # Cannot determine hand

        hand = str(hand).strip().lower()
        side_to_keep = None
        if hand == 'left': side_to_keep = 'Right'
        elif hand == 'right': side_to_keep = 'Left'
        else: return row # Unrecognized hand condition

        # Create a dictionary from the row for easier modification
        row_dict = row.to_dict()
        for key_base in imaging_keys_base:
            for version in imaging_versions:
                # Column name in the original DatScan data (now potentially with _DatScan suffix if not merged cleanly)
                source_col_name = f"{key_base}_{side_to_keep}_{version}"
                source_col_name_suffixed = f"{source_col_name}_DatScan" # Check for suffixed version too

                # Construct the new generic 'Contralateral' column name
                new_col_name = f"Contralateral_{key_base}_{version}"

                # Get value: prioritize non-suffixed, then suffixed
                value_to_assign = np.nan # Default to NaN
                if source_col_name in row_dict:
                    value_to_assign = row_dict.get(source_col_name, np.nan)
                elif source_col_name_suffixed in row_dict:
                     value_to_assign = row_dict.get(source_col_name_suffixed, np.nan)

                # Assign the value to the new column in the dictionary
                row_dict[new_col_name] = value_to_assign

        return pd.Series(row_dict) # Return modified row as a Series

    print("Applying contralateral selection logic...")
    merged_df = merged_df.apply(select_contralateral, axis=1)

    # Identify columns to drop (original L/R specific and potentially suffixed ones)
    cols_to_drop = []
    for key_base in imaging_keys_base:
        for version in imaging_versions:
            for side in ['Left', 'Right']:
                 col_name = f"{key_base}_{side}_{version}"
                 cols_to_drop.append(col_name)
                 cols_to_drop.append(f"{col_name}_DatScan") # Add suffixed version
            # Add mean columns if they exist
            for mean_prefix in ["Mean_", "Mean"]: # Handle variations
                 mean_col_name = f"{mean_prefix}{key_base.lower()}_{version}"
                 cols_to_drop.append(mean_col_name)
                 cols_to_drop.append(f"{mean_col_name}_DatScan")


    # Drop only columns that actually exist in the dataframe
    cols_to_drop_existing = [col for col in cols_to_drop if col in merged_df.columns]
    if cols_to_drop_existing:
        # print(f"Dropping original side-specific/mean DatScan columns: {cols_to_drop_existing}") # Less verbose
        merged_df.drop(columns=cols_to_drop_existing, inplace=True, errors='ignore')
    # else: print("No original side-specific/mean DatScan columns found to drop.")


    print(f"Shape after dropping original DatScan columns: {merged_df.shape}")
    # Display columns related to contralateral selection for verification
    contra_cols = [c for c in merged_df.columns if c.startswith('Contralateral_')]
    print(f"Contralateral columns created: {contra_cols}")
    if not contra_cols: print("Warning: No contralateral columns seem to have been created!")


    return merged_df

# ========================
# Main Execution Block
# ========================
if __name__ == '__main__':
    print("--- Starting Preprocessing Script ---")
    # Define base paths for the two kinematic tasks relative to script location
    fingertapping_path = os.path.join(input_folder_path, 'Fingertapping')
    hand_movements_path = os.path.join(input_folder_path, 'Hand_Movements')

    # Process kinematic data
    ft_df = process_kinematic_subfolder(fingertapping_path, "ft")
    hm_df = process_kinematic_subfolder(hand_movements_path, "hm")

    # --- Merge Kinematic Data ---
    if ft_df.empty and hm_df.empty:
        print("\nBoth ft_df and hm_df are empty after processing. Cannot proceed.")
        kinematic_summary_df = pd.DataFrame()
    elif ft_df.empty:
        print("\nft_df is empty. Using only hm_df.")
        kinematic_summary_df = hm_df
    elif hm_df.empty:
        print("\nhm_df is empty. Using only ft_df.")
        kinematic_summary_df = ft_df
    else:
        print("\nMerging Finger Tapping and Hand Movement data...")
        merge_keys = ["Patient ID", "Date of Visit", "Medication Condition"]
        try:
            kinematic_summary_df = pd.merge(ft_df, hm_df, on=merge_keys, how="outer")
            print(f"Shape after kinematic merge: {kinematic_summary_df.shape}")
            if not kinematic_summary_df.empty:
                print("Medication Conditions in merged kinematic summary:", kinematic_summary_df['Medication Condition'].value_counts())
        except Exception as e_merge_k:
            print(f"Error merging kinematic dataframes: {e_merge_k}")
            kinematic_summary_df = pd.DataFrame() # Ensure it's empty on error


    # Proceed only if kinematic_summary_df is not empty
    if not kinematic_summary_df.empty:
        print("\n--- Processing Merged Kinematic Data ---")
        # --- Standardization and Cleaning ---
        def convert_pid_final(pid):
            try: return str(int(float(pid)))
            except (ValueError, TypeError): return str(pid).strip()
        kinematic_summary_df['Patient ID'] = kinematic_summary_df['Patient ID'].apply(convert_pid_final)

        # Convert 'Date of Visit' to datetime for sorting/calculations
        kinematic_summary_df['Date of Visit DT'] = pd.to_datetime(kinematic_summary_df['Date of Visit'], format='%d.%m.%Y', errors='coerce')
        rows_before_dropna_date = len(kinematic_summary_df)
        kinematic_summary_df.dropna(subset=['Date of Visit DT'], inplace=True)
        if len(kinematic_summary_df) < rows_before_dropna_date:
            print(f"Dropped {rows_before_dropna_date - len(kinematic_summary_df)} rows with invalid 'Date of Visit'.")

        # Remove potential duplicate rows
        key_cols = ["Patient ID", "Date of Visit", "Medication Condition"]
        # Add hand conditions dynamically if they exist
        if "ft_Hand Condition" in kinematic_summary_df.columns: key_cols.append("ft_Hand Condition")
        if "hm_Hand Condition" in kinematic_summary_df.columns: key_cols.append("hm_Hand Condition")
        initial_rows_dedup = len(kinematic_summary_df)
        kinematic_summary_df.drop_duplicates(subset=key_cols, keep='first', inplace=True)
        if len(kinematic_summary_df) < initial_rows_dedup:
            print(f"Dropped {initial_rows_dedup - len(kinematic_summary_df)} duplicate rows based on keys: {key_cols}")

        # Calculate 'Days Since First Visit'
        kinematic_summary_df.sort_values(by=['Patient ID', 'Date of Visit DT'], inplace=True)
        # Ensure groupby transform handles potential empty groups or all NaT dates gracefully
        kinematic_summary_df['Days Since First Visit'] = kinematic_summary_df.groupby('Patient ID')['Date of Visit DT'].transform(
             lambda x: (x - x.min()).dt.days if pd.notna(x.min()) else np.nan
        )

        # --- Rearrange columns ---
        cols_order = ['Patient ID', 'Date of Visit', 'Medication Condition', 'Days Since First Visit']
        # Add kinematic metadata cols
        meta_cols = [c for c in kinematic_summary_df.columns if ('Hand Condition' in c or 'Kinematic Task' in c) and c not in cols_order]
        cols_order.extend(sorted(meta_cols))
        # Add ft kinematic data cols
        ft_data_cols = [c for c in kinematic_summary_df.columns if c.startswith('ft_') and c not in cols_order]
        cols_order.extend(sorted(ft_data_cols))
        # Add hm kinematic data cols
        hm_data_cols = [c for c in kinematic_summary_df.columns if c.startswith('hm_') and c not in cols_order]
        cols_order.extend(sorted(hm_data_cols))
        # Add remaining cols (should include Date of Visit DT and others)
        remaining_cols = [c for c in kinematic_summary_df.columns if c not in cols_order]
        cols_order.extend(remaining_cols)

        # Ensure Date of Visit DT is removed before final selection if present
        if 'Date of Visit DT' in cols_order: cols_order.remove('Date of Visit DT')
        # Ensure all original columns are accounted for to prevent accidental dropping
        final_cols = [c for c in cols_order if c in kinematic_summary_df.columns]
        kinematic_summary_df = kinematic_summary_df[final_cols].copy() # Use copy

        # --- Load DatScan and Merge ---
        dat_scan_df = load_dat_scan(input_folder_path)

        # Define output path for the final merged file
        output_base_name = "merged_summary_with_medon.csv"
        final_output_path = os.path.join(input_folder_path, output_base_name)

        if dat_scan_df is not None and not dat_scan_df.empty:
            # Perform the merge only if DatScan data is valid
            merged_df = merge_dat_scan(kinematic_summary_df, dat_scan_df)
            # Save the merged result
            try:
                merged_df.to_csv(final_output_path, index=False, sep=';', decimal='.')
                print(f"\nMerged summary (Kinematic + Imaging) saved to: {final_output_path}")
                print(f"Final DataFrame shape: {merged_df.shape}")
                if 'Medication Condition' in merged_df.columns:
                    print("Final Medication Conditions:", merged_df['Medication Condition'].value_counts())
            except Exception as e_save:
                print(f"Failed to save merged summary CSV: {e_save}")
        else:
             # If DatScan failed, save the kinematic-only summary
             print("\nDatScan data could not be loaded or was empty. Saving kinematic summary only.")
             try:
                kinematic_summary_df.to_csv(final_output_path, index=False, sep=';', decimal='.')
                print(f"Kinematic-only summary saved to: {final_output_path}")
                print(f"Kinematic summary shape: {kinematic_summary_df.shape}")
                if 'Medication Condition' in kinematic_summary_df.columns:
                     print("Final Medication Conditions:", kinematic_summary_df['Medication Condition'].value_counts())
             except Exception as e_save_k:
                print(f"Failed to save kinematic-only summary CSV: {e_save_k}")

    else:
        print("\nKinematic summary DataFrame is empty after processing subfolders. Cannot create output files.")

print("\n--- Preprocessing script execution finished ---")
# --- END OF FILE ---