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

def process_input_folder(folder_path):
    summary_list = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue
        
        # Skip any summary files if present
        if filename in ['summary.csv', 'merged_summary.csv']:
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
        
        # Detect separator for the kinematic CSV file
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
        except Exception as e:
            print(f"Skipping file {filename}: could not read or process CSV file")
            continue
        
        # Create a summary row for this file
        summary_row = {
            'Patient ID': patient_id,
            'Date of Visit': date_of_visit_str,
            'Medication Condition': med_condition,
            'Kinematic Task': kinematic_task,
            'Hand Condition': hand_condition
        }
        summary_row.update(variables)
        summary_list.append(summary_row)
    
    # Create a DataFrame from the summary list
    summary_df = pd.DataFrame(summary_list)
    
    if 'Date of Visit' in summary_df.columns:
        # Standardize Patient ID: convert to int then to string (e.g., "078" becomes "78")
        def convert_pid(pid):
            try:
                return str(int(pid))
            except Exception as e:
                return pid
        summary_df['Patient ID'] = summary_df['Patient ID'].apply(convert_pid)
        
        # Debug: print kinematic Patient IDs
        print("Kinematic Patient IDs:", summary_df['Patient ID'].unique())
        
        # Convert 'Date of Visit' to datetime for sorting and calculation
        summary_df['Date of Visit'] = pd.to_datetime(summary_df['Date of Visit'], format='%d.%m.%Y')
        summary_df.drop_duplicates(inplace=True)
        
        # Calculate 'Days Since First Visit' for each patient
        summary_df['Days Since First Visit'] = summary_df.groupby('Patient ID')['Date of Visit'].transform(
            lambda x: (x - x.min()).dt.days)
        
        # Rearrange columns to position 'Days Since First Visit' after 'Medication Condition'
        cols = list(summary_df.columns)
        if 'Days Since First Visit' in cols:
            cols.remove('Days Since First Visit')
        try:
            insert_at = cols.index('Medication Condition') + 1
        except ValueError:
            insert_at = len(cols)
        cols.insert(insert_at, 'Days Since First Visit')
        summary_df = summary_df[cols]
        
        # Convert 'Date of Visit' back to string format for saving
        summary_df['Date of Visit'] = summary_df['Date of Visit'].dt.strftime('%d.%m.%Y')
        
        # Save the kinematic summary DataFrame to a CSV file
        summary_file_path = os.path.join(folder_path, 'summary.csv')
        try:
            summary_df.to_csv(summary_file_path, index=False)
            print(f"Kinematic summary CSV file created at: {summary_file_path}")
        except Exception as e:
            print(f"Failed to save kinematic summary CSV at {summary_file_path}: {e}")
    else:
        print("The 'Date of Visit' column is missing from the summary DataFrame. Unable to proceed with sorting and saving the summary.")
    
    return summary_df

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
    
    # Merge on 'Patient ID'
    merged_df = pd.merge(kinematic_df, dat_scan_df, on="Patient ID", how="left")
    
    # Define which imaging columns are side-specific.
    imaging_keys = ['Striatum', 'Putamen', 'Caudate']
    
    def select_contralateral(row):
        hand = row.get('Hand Condition', '').strip().lower()
        if hand == 'left':
            side = 'Right'
        elif hand == 'right':
            side = 'Left'
        else:
            # If hand condition is unknown, leave imaging data as missing.
            return row
        
        # For each imaging key and for each software version, keep only the contralateral side data.
        for version in ['old', 'Z', 'new']:
            for key in imaging_keys:
                col_name = f"{key}_{side}_{version}"
                new_col_name = f"Contralateral_{key}_{version}"
                row[new_col_name] = row.get(col_name, None)
        return row

    merged_df = merged_df.apply(select_contralateral, axis=1)
    
    # Optionally, drop the original imaging columns if you want only the contralateral ones in the final output.
    cols_to_drop = [col for col in merged_df.columns 
                    if any(tag in col for tag in ['_old', '_Z', '_new']) and not col.startswith("Contralateral_")]
    merged_df.drop(columns=cols_to_drop, inplace=True)
    
    return merged_df

if __name__ == '__main__':
    # Process kinematic files to generate a summary DataFrame.
    kinematic_summary_df = process_input_folder(input_folder_path)

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
