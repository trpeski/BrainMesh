base_path = '/content/'

def set_base_path (new_base_path) :
  base_path = new_base_path

def get_base_path () :
  return base_path

def get_subject_directory (subject_p) :
  return f'subject_{subject_p}'

def get_data_path (subject_p, recording_id_p) :
  return get_subject_directory(subject_p)+f'/recording_{recording_id_p}.edf'

def get_patient_seizures_path (subject_p) :
  return get_subject_directory(subject_p)+f'/seizures.edf'

def count_word_occurrences(file_path, word):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += line.count(word)
    return count
