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
    with open(file_path, 'r') as file:
        sample = file.read(1024)  # Read a small sample of the file
        sniffer = csv.Sniffer()
        detected_separator = sniffer.sniff(sample).delimiter
        return detected_separator

def parse_kinematic_filename(filename):
    """
    Expected format:
      DateOfVisit_PatientID_MedCondition_KinematicTask - HandCondition.csv
    Example:
      2016-10-17_078_Off_Hand movement - Left.csv
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    if len(parts) < 4:
        print(f"Skipping file {filename}: unexpected filename format")
        return None
    
    date_of_visit = parts[0]
    patient_id = parts[1]
    med_condition = parts[2]
    
    # The fourth part is expected to contain the kinematic task and hand condition,
    # separated by " - "
    kinematic_task = ""
    hand_condition = ""
    if ' - ' in parts[3]:
        task_parts = parts[3].split(' - ')
        if len(task_parts) == 2:
            kinematic_task = task_parts[0].strip()
            hand_condition = task_parts[1].strip()
        else:
            kinematic_task = parts[3].strip()
    else:
        kinematic_task = parts[3].strip()
    
    return date_of_visit, patient_id, med_condition, kinematic_task, hand_condition

def process_kinematic_subfolder(folder_path, prefix):
    """
    Process all CSV files in a specific subfolder.
    The function leaves the common keys unprefixed (Patient ID, Date of Visit, Medication Condition)
    but renames task-specific metadata and all kinematic variable columns with the given prefix.
    """
    summary_list = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue
        
        file_path = os.path.join(folder_path, filename)
        parsed = parse_kinematic_filename(filename)
        if parsed is None:
            continue
        
        date_of_visit, patient_id, med_condition, kinematic_task, hand_condition = parsed
        
        # Convert date from YYYY-MM-DD to DD.MM.YYYY
        try:
            date_of_visit_dt = pd.to_datetime(date_of_visit, format='%Y-%m-%d')
            date_of_visit_str = date_of_visit_dt.strftime('%d.%m.%Y')
        except Exception as e:
            print(f"Skipping file {filename}: could not parse date {date_of_visit}")
            continue
        
        # Detect separator for the CSV file
        try:
            file_separator = detect_separator(file_path)
        except Exception as e:
            print(f"Skipping file {filename}: error detecting separator")
            continue
        
        try:
            data_df = pd.read_csv(file_path, sep=file_separator, header=None)
            data_df.columns = ['Attribute', 'Value']
            # Lowercase the attribute names for consistency
            data_df['Attribute'] = data_df['Attribute'].str.lower()
            variables = data_df.set_index('Attribute')['Value'].to_dict()
            
            # Add prefix to each kinematic variable key
            variables = {f"{prefix}_{k}": v for k, v in variables.items()}
        except Exception as e:
            print(f"Skipping file {filename}: could not read or process CSV file")
            continue
        
        # Create summary row with common keys unprefixed
        summary_row = {
            'Patient ID': patient_id,
            'Date of Visit': date_of_visit_str,
            'Medication Condition': med_condition,
            # For task-specific metadata, add the prefix
            f"{prefix}_Kinematic Task": kinematic_task,
            f"{prefix}_Hand Condition": hand_condition
        }
        summary_row.update(variables)
        summary_list.append(summary_row)
    
    return pd.DataFrame(summary_list)

def load_dat_scan(input_folder):
    """
    Load and preprocess the DatScan.csv file.
    Renames columns, converts the date column, drops the Gender column,
    and handles non-finite Patient IDs.
    """
    dat_scan_path = os.path.join(input_folder, 'DatScan.csv')
    try:
        dat_scan_df = pd.read_csv(dat_scan_path, delimiter=';')
    except Exception as e:
        raise Exception(f"Error reading DatScan.csv: {e}")
    
    # Rename columns for clarity, only including new software and Z-scores.
    rename_dict = {
        "No.": "Patient ID",
        "Date of Scan (DOS)": "Date of Scan",
        "Gender       0 = female\n1= male": "Gender",
        
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
    
    # Remove rows with non-finite or missing Patient IDs to avoid conversion errors
    dat_scan_df = dat_scan_df.dropna(subset=['Patient ID'])
    dat_scan_df = dat_scan_df[np.isfinite(dat_scan_df['Patient ID'])]
    
    # Convert 'Patient ID' to int then to string
    dat_scan_df['Patient ID'] = dat_scan_df['Patient ID'].astype(int).astype(str)
    
    # Drop the Gender column as it is not needed.
    if 'Gender' in dat_scan_df.columns:
        dat_scan_df.drop(columns=['Gender'], inplace=True)
    
    # Debug print: Show all patient IDs loaded from DatScan.csv.
    print("Debug: Loaded DatScan.csv. Patient IDs:", dat_scan_df['Patient ID'].tolist())
    
    return dat_scan_df

def merge_dat_scan(kinematic_df, dat_scan_df):
    """
    Merge the kinematic summary with DatScan imaging data.
    For each row, only the contralateral imaging data (based on Hand Condition) are retained.
    """
    # Ensure 'Patient ID' in kinematic_df is also standardized as string
    kinematic_df['Patient ID'] = kinematic_df['Patient ID'].astype(str)
    
    # Merge on common key(s). Here we merge on Patient ID. If you wish to be more granular (e.g. by Date of Visit),
    # you can include that in the keys.
    merged_df = pd.merge(kinematic_df, dat_scan_df, on="Patient ID", how="left")
    
    # Define which imaging keys are side-specific.
    imaging_keys = ['Striatum', 'Putamen', 'Caudate']
    
    def select_contralateral(row):
        # Determine hand condition. If both tasks are available, prefer the Fingertapping one.
        hand = None
        if f"ft_Hand Condition" in row and pd.notna(row[f"ft_Hand Condition"]):
            hand = row[f"ft_Hand Condition"]
        elif f"hm_Hand Condition" in row and pd.notna(row[f"hm_Hand Condition"]):
            hand = row[f"hm_Hand Condition"]
        else:
            hand = row.get('Hand Condition', '')
        
        hand = str(hand).strip().lower()
        if hand == 'left':
            side = 'Right'
        elif hand == 'right':
            side = 'Left'
        else:
            return row
        
        # For each imaging key and for each relevant software version, keep only the contralateral side data.
        for version in ['Z', 'new']:
            for key in imaging_keys:
                col_name = f"{key}_{side}_{version}"
                new_col_name = f"Contralateral_{key}_{version}"
                row[new_col_name] = row.get(col_name, None)
        return row

    merged_df = merged_df.apply(select_contralateral, axis=1)
    
    # Optionally, drop the original imaging columns if you want only the contralateral ones in the final output.
    cols_to_drop = [col for col in merged_df.columns 
                    if any(tag in col for tag in ['_Z', '_new']) and not col.startswith("Contralateral_")]
    merged_df.drop(columns=cols_to_drop, inplace=True)
    
    return merged_df

if __name__ == '__main__':
    # Define paths for the two kinematic subfolders
    fingertapping_path = os.path.join(input_folder_path, 'Fingertapping')
    hand_movements_path = os.path.join(input_folder_path, 'Hand_Movements')
    
    # Process each subfolder with its respective prefix.
    # The common keys (Patient ID, Date of Visit, Medication Condition) remain unprefixed.
    ft_df = process_kinematic_subfolder(fingertapping_path, "ft")
    hm_df = process_kinematic_subfolder(hand_movements_path, "hm")
    
    # Merge the two kinematic DataFrames on common keys so that findings for the same patient/visit are in one row.
    # Here we merge on "Patient ID", "Date of Visit", and "Medication Condition".
    kinematic_summary_df = pd.merge(ft_df, hm_df, on=["Patient ID", "Date of Visit", "Medication Condition"], how="outer")
    
    # Standardize Patient ID: convert to int then to string (e.g., "078" becomes "78")
    def convert_pid(pid):
        try:
            return str(int(pid))
        except Exception as e:
            return pid
    if 'Patient ID' in kinematic_summary_df.columns:
        kinematic_summary_df['Patient ID'] = kinematic_summary_df['Patient ID'].apply(convert_pid)
    
    # Debug: print kinematic Patient IDs
    print("Kinematic Patient IDs:", kinematic_summary_df['Patient ID'].unique())
    
    # Convert 'Date of Visit' to datetime for sorting and calculation
    kinematic_summary_df['Date of Visit'] = pd.to_datetime(kinematic_summary_df['Date of Visit'], format='%d.%m.%Y')
    kinematic_summary_df.drop_duplicates(inplace=True)
    
    # Calculate 'Days Since First Visit' for each patient
    kinematic_summary_df['Days Since First Visit'] = kinematic_summary_df.groupby('Patient ID')['Date of Visit'].transform(
        lambda x: (x - x.min()).dt.days)
    
    # Rearrange columns to position 'Days Since First Visit' after 'Medication Condition'
    cols = list(kinematic_summary_df.columns)
    if 'Days Since First Visit' in cols:
        cols.remove('Days Since First Visit')
    try:
        insert_at = cols.index('Medication Condition') + 1
    except ValueError:
        insert_at = len(cols)
    cols.insert(insert_at, 'Days Since First Visit')
    kinematic_summary_df = kinematic_summary_df[cols]
    
    # Convert 'Date of Visit' back to string format for saving
    kinematic_summary_df['Date of Visit'] = kinematic_summary_df['Date of Visit'].dt.strftime('%d.%m.%Y')
    
    # Save the kinematic summary DataFrame to a CSV file
    summary_file_path = os.path.join(input_folder_path, 'summary.csv')
    try:
        kinematic_summary_df.to_csv(summary_file_path, index=False)
        print(f"Kinematic summary CSV file created at: {summary_file_path}")
    except Exception as e:
        print(f"Failed to save kinematic summary CSV at {summary_file_path}: {e}")
    
    # Load DatScan data.
    try:
        dat_scan_df = load_dat_scan(input_folder_path)
    except Exception as e:
        print(f"Error loading DatScan data: {e}")
        dat_scan_df = None
    
    if dat_scan_df is not None:
        # Merge kinematic summary with DatScan data (contralateral imaging columns).
        merged_df = merge_dat_scan(kinematic_summary_df, dat_scan_df)
            
        print("Kinematic Patient IDs:", kinematic_summary_df['Patient ID'].unique())
        print("DatScan Patient IDs:", dat_scan_df['Patient ID'].unique())
        
        # Save the merged summary DataFrame to a new CSV file in the same folder.
        merged_summary_file_path = os.path.join(input_folder_path, 'merged_summary.csv')
        try:
            merged_df.to_csv(merged_summary_file_path, index=False)
            print(f"Merged summary CSV file created at: {merged_summary_file_path}")
        except Exception as e:
            print(f"Failed to save merged summary CSV at {merged_summary_file_path}: {e}")
