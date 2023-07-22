def get_subject_directory (subject_p) :
  return f'subject_{subject_p}'

def get_data_path (subject_p, recording_id_p) :
  return get_subject_directory(subject_p)+f'/recording_{recording_id_p}.edf'

def get_patient_seizures_path (subject_p) :
  return get_subject_directory(subject_p)+f'/seizures.edf'
