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
