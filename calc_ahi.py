# This Python file uses the following encoding: utf-8
"""
Created on Thu Apr 16 10:45:09 2020

@author: umaer
"""

# Import necessary libraries
import pandas as pd 
import numpy as np
import datetime as dt
import glob
import zipfile
# Disable chained assignment warnings in pandas
pd.options.mode.chained_assignment = None

# Define a function to verify Apnea-Hypopnea Index (AHI) from sleep study data
def verify_ahi(files_path, demographics_path, cpap_path): 
    patient_info = []  # List to store patient data summaries
    i = 0  # Initialize a counter for tracking the number of processed patients
    files = glob.glob(files_path + '*.csv')  # Retrieve all CSV files in the specified directory
    df_cpap = pd.read_excel(cpap_path)  # Load CPAP data from an Excel file

    # Loop through each file in the files list
    for file_name in files:
        subject_code = file_name[-13:-4]  # Extract subject code from the file name
        # Load CSV file, ignoring bad lines, and using only the first three columns
        df = pd.read_csv(file_name, error_bad_lines=False, usecols=[0,1,2])
        
        # Check if the subject is listed in the CPAP data
        if subject_code in df_cpap['Subjects'].values:
            # Find the index of the subject in the CPAP dataframe
            j = int(df_cpap.index[df_cpap['Subjects'] == subject_code].to_numpy())
            # Check if a split night is mentioned and filter the dataframe accordingly
            split = df['Event'].str.contains(df_cpap['Split Night'][j], case=False)
            split_idx = df['Event'][split].index[0]
            df = df.loc[:split_idx]
            
        # Filter the dataframe for sleep stages and correct any erroneous durations
        stages = df.loc[df['Event'].isin([' Stage1', ' Stage2', ' Stage3', ' REM'])]
        stages['Duration (seconds)'] = stages['Duration (seconds)'].replace(2592000, 30)
        stages['Duration (seconds)'] = stages['Duration (seconds)'].replace(0, 30)        
        sleep_time = stages['Duration (seconds)'].sum()  # Calculate total sleep time
        
        events = pd.Index(df['Event'])  # Create an index of events
        
        # Calculate statistics for various sleep events (e.g., ObstructiveApnea, CentralApnea, etc.)
        # This includes counting occurrences and averaging durations for each event type.
        # If an event does not occur, set its count and average duration to NaN or 0 accordingly.
        # For each event type, check its presence, calculate the count, and mean duration.
        # Repeat this process for ObstructiveApnea, CentralApnea, MixedApnea, Hypopnea
        if ' ObstructiveApnea' in events: 
            obs_idx = events.get_loc(' ObstructiveApnea')
            obs_events = df.iloc[obs_idx]
            n_obs = len(obs_events)
            obs_events_dur = obs_events['Duration (seconds)']
            mean_obs_events_dur = obs_events_dur.mean()
        else:
            mean_obs_events_dur = np.nan
            n_obs = 0
            
        if ' CentralApnea' in events: 
            cen_idx = events.get_loc(' CentralApnea')
            cen_events = df.iloc[cen_idx]
            n_cen = len(cen_events)
            cen_events_dur = cen_events['Duration (seconds)']
            mean_cen_events_dur = cen_events_dur.mean()
        else:
            mean_cen_events_dur = np.nan
            n_cen = 0
            
        if ' MixedApnea' in events:
            mix_idx = events.get_loc(' MixedApnea')
            mix_events = df.iloc[mix_idx]
            n_mix = len(mix_events)
            mix_events_dur = mix_events['Duration (seconds)']
            mean_mix_events_dur = mix_events_dur.mean()
        else:
            mean_mix_events_dur = np.nan
            n_mix = 0
                        
        if ' Hypopnea' in events:
            hyp_idx = events.get_loc(' Hypopnea')
            hyp_events = df.iloc[hyp_idx]
            n_hyp = len(hyp_events)
            hyp_events_dur = hyp_events['Duration (seconds)']
            mean_hyp_events_dur = hyp_events_dur.mean()
        else:
            mean_hyp_events_dur = np.nan
            n_hyp = 0
        
        # Calculate the Apnea-Hypopnea Index (AHI) for the subject
        ahi = (n_obs + n_mix + n_hyp + n_cen)/(sleep_time/3600)
    
        # Compile the patient summary and add it to the patient_info list
        summary = {'s_code': subject_code, 'n_obs': n_obs, 'n_cen': n_cen, 'n_mix': n_mix,
                   'n_hyp': n_hyp, 'ahi': ahi}
        patient_info.append(summary)
        i += 1  # Increment the patient counter
        print('Patient {} out of {}'.format(i, len(files)))  # Print progress
        
    # Convert the patient_info list to a DataFrame and clean up the data
    df_events = pd.DataFrame(patient_info)
    df_events = df_events.dropna(subset=['ahi'])  # Drop rows where AHI is NaN
    df_events = df_events[df_events['ahi'] != np.inf]  # Remove infinite values
    df_events = df_events[df_events['ahi'] < 200]  # Filter out unreasonably high AHI values
    # Save the cleaned DataFrame to an Excel file
    df_events.to_excel('C:/Users/umaer/OneDrive/Desktop/AHI.xlsx', float_format='%.1f')
    
    return df_events  # Return the DataFrame containing patient AHI summaries

# Define file paths for the sleep study data, demographics information, and CPAP data
files_path = 'E:/EDFs/'
demographics_path = 'C:/Users/umaer/OneDrive/Documents/PhD/Journal Papers/OSA Prediction/Data/20191106DemographicSTAGES.xlsx'
cpap_path = 'C:/Users/umaer/OneDrive/Documents/PhD/Journal Papers/OSA Prediction/Data/CPAP.xlsx'

# Call the verify_ahi function with the specified file paths to process the data and verify AHI values
demographics = verify_ahi(files_path, demographics_path, cpap_path)
