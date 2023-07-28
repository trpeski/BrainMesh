import requests
import os

url = 'https://physionet.org/files/chbmit/1.0.0/'

def load_edf_data(subject_p,recording_id_p):
  FILE = 'chb'+subject_p+'_'+recording_id_p+'.edf' # <-- choose file

  filename = 'chb'+subject_p+'/' + FILE + '?download'

  new_filename = get_data_path(subject_p, recording_id_p)
  if os.path.isfile(new_filename) :
    return

  if not (os.path.exists(get_subject_directory(subject_p))) :
    os.mkdir(get_subject_directory(subject_p))

  response = requests.get(url + filename)

  with open(new_filename, 'wb') as f:
      f.write(response.content)

def load_patient_seizures (subject_p) :
  new_filename_s = get_patient_seizures_path(subject_p)
  if not (os.path.exists(get_subject_directory(subject_p))) :
    os.mkdir(get_subject_directory(subject_p))

  FILE_S = 'chb'+subject_p+'-summary.txt' # <-- load seizures start and end times
  filename_s = 'chb'+subject_p+'/' + FILE_S +'?download'
  response_s = requests.get(url + filename_s)
  with open(new_filename_s, 'wb') as f:
      f.write(response_s.content)
