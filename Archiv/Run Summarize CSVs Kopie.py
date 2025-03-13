import os
import re
import pandas as pd
import csv

# Get the current script directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder containing all CSV files
input_folder_path = os.path.join(script_dir, 'Input')

# Function to detect the separator of a CSV file
def detect_separator(file_path):
    with open(file_path, 'r') as file:
        sample = file.read(1024)  # Read a small sample of the file
        sniffer = csv.Sniffer()
        detected_separator = sniffer.sniff(sample).delimiter
        return detected_separator

# New filename parser for the kinematic files based on the new format:
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

# Function to process all kinematic CSV files in the input folder
def process_input_folder(folder_path):
    summary_list = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue
        
        # Skip any summary file if present
        if filename == 'summary.csv':
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
        file_separator = detect_separator(file_path)
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
        
        # Save the summary DataFrame to a new CSV file in the same folder
        summary_file_path = os.path.join(folder_path, 'summary.csv')
        try:
            summary_df.to_csv(summary_file_path, index=False)
            print(f"Summary CSV file created at: {summary_file_path}")
        except Exception as e:
            print(f"Failed to save summary CSV at {summary_file_path}: {e}")
    else:
        print("The 'Date of Visit' column is missing from the summary DataFrame. Unable to proceed with sorting and saving the summary.")

# Run the summary operation on the input folder
process_input_folder(input_folder_path)
