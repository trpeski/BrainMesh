
"""Helper funcitons to work with paths"""

base_path = 'D:/GD3iz/'


def set_base_path (new_base_path) :
  base_path = new_base_path

def get_base_path () :
  return base_path



# Get filenames

def get_recording_file_name (recording_id, file_type) :
  return f'recording_{recording_id}.{file_type}'

def get_patient_summary_file_name (file_type) :
  return f'seizures.{file_type}'

def get_seizure_seconds_file_name () :
  return f'seizure_seconds.pickle'

def get_seizure_frames_file_name () :
  return f'seizure_frames.pickle'


# Get relative path

def get_subject_directory (subject_p) :
  return f'subject_{subject_p}'

def get_data_path (subject_p, recording_id_p, file_type) :
  return get_subject_directory(subject_p)+f'/recording_{recording_id_p}.{file_type}'

def get_patient_summary_path (subject_p, file_type='edf') :
  return get_subject_directory(subject_p)+f'/seizures.{file_type}'

def get_seizure_seconds_path (subject_p) :
  return get_subject_directory(subject_p)+f'/seizure_seconds.pickle'

def get_seizure_frames_path (subject_p) :
  return get_subject_directory(subject_p)+f'/seizure_frames.pickle'


# Load pickle

def load_pickle (path) :
  with open(path, 'rb') as handle :
    return pickle.load(handle)

def save_pickle (path, data) :
  with open(path, 'wb') as handle :
    pickle.dump(data, path)
