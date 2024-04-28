#Parameters

interval_in_seconds = 15
interval_step_p_in_seconds = 3
# If step < length, the intervals overlap
interval_length_p = interval_in_seconds * 256
interval_step_p = interval_step_p_in_seconds * 256

def set_interval_seconds(s, step):
  interval_length_p = s * 256
  interval_step_p = step * 256

  if interval_step_p > interval_length_p :
    print('Invalid interval step-length combination.')
    print('Interval step cannot be larger than length, as that would lead to frames being skipped.')



#def extract_seizure_times(file_name, recording):
def extract_seizure_times(subject_id, recording_id, source_path):
    file_name = get_patient_summary_path(subject_id, 'edf')

    with open(source_path + file_name, 'r', encoding='latin-1') as f:
      content = f.read()

    file_info = content.split("\n\n")
    for info in file_info:
        if f'_{recording_id}.edf' in info:
            print(recording_id)
            seizure_times = []
            lines = info.split("\n")
            for line in lines:
                print(line)
                if ("Start Time" in line) and ("Seizure" in line):
                    seizure_start = int(line.split(":")[1].strip().split(" ")[0])
                    seizure_times.append([seizure_start])
                elif ("End Time" in line) and ("Seizure" in line):
                    seizure_end = int(line.split(":")[1].strip().split(" ")[0])
                    seizure_times[-1].append(seizure_end)

            if len(seizure_times) == 0:
                print("No seizures found in the file.")
                return []

            return seizure_times

    print("File not found.")
    return []


"""Script to download the data for a subject and recording"""

import requests
import os

url = 'https://physionet.org/files/chbmit/1.0.0/'

def load_edf_data(subject_p,recording_id_p, target_path):
  FILE = 'chb'+subject_p+'_'+recording_id_p+'.edf'
  filename = 'chb'+subject_p+'/' + FILE + '?download'

  new_filename = get_data_path(subject_p, recording_id_p, 'edf')
  if os.path.isfile(target_path+new_filename) :
    return new_filename

  if not (os.path.exists(target_path+get_subject_directory(subject_p))) :
    os.makedirs(target_path+get_subject_directory(subject_p))

  print('requesting')
  response = requests.get(url + filename)
  if response.status_code == 404 :
    print(f"File '{filename}' does not exist in database")
    return None
  print('acquired')

  with open(target_path+new_filename, 'wb') as f:
      f.write(response.content)

  return new_filename

"""Load the summary file"""

def load_patient_seizures (subject_p, target_path) :
  new_filename_s = get_patient_summary_path(subject_p)

  if not (os.path.exists(target_path+get_subject_directory(subject_p))) :
    os.makedirs(target_path+get_subject_directory(subject_p))

  FILE_S = 'chb'+subject_p+'-summary.txt'
  filename_s = 'chb'+subject_p+'/' + FILE_S +'?download'
  response_s = requests.get(url + filename_s)
  with open(target_path+new_filename_s, 'wb') as f:
      f.write(response_s.content)

"""Extract the data from EDF into pandas dataframe format"""

import mne
import pandas as pd

def load_dataframe(filename):
  # Load the EDF file into an MNE Raw object
  raw = mne.io.read_raw_edf(filename)  # CHANGE PATH FOR YOUR CASE

  # Extract signal data and save to CSV file
  data, times = raw[:, :]
  ch = raw.ch_names
  ch = ','.join(ch)
  data = pd.DataFrame(data.T, columns=ch.split(','))
  
  
  chann = raw.ch_names
  print(chann)
  channels = [
    'FP1-F7',
    'F7-T7',
    'T7-P7',
    'P7-O1',
    'FP1-F3',
    'F3-C3',
    'C3-P3',
    'P3-O1',
    'FP2-F4',
    'F4-C4',
    'C4-P4',
    'P4-O2',
    'FP2-F8',
    'F8-T8',
    'T8-P8-0',
    'P8-O2',
    'FZ-CZ',
    'CZ-PZ',
    'P7-T7',
    'T7-FT9',
    'FT9-FT10',
    'FT10-T8',
    'T8-P8-1'
      
  ]
  channels.sort()
  a = np.array(channels)
  b = np.array(chann)
  if not (np.isin(a, b)[0] or np.isin(a, b)[1] or np.isin(a, b)[2] or np.isin(a, b)[3] or np.isin(a, b)[4] or np.isin(a, b)[5] or np.isin(a, b)[22]):
    channels = chann[:23]
 
  try:
    data = data[channels]
  except:
    channels = chann[:23]
 

 
  return (data, channels)

"""Count word occurrences helper function to be used to load seizures from the summary"""

def count_word_occurrences(file_path, word):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += line.count(word)
    return count

"""Helper functions to change seconds into frames, taking into account that all recordings are 256Hz"""

def seconds_to_frames(seconds):
  return [[256*s for s in sec] for sec in seconds]
