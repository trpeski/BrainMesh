

"""Compute the Hilbert envelopes of given data for every channel"""

import matplotlib.pyplot as plt
from scipy.signal import hilbert

def compute_envelopes(data):
  envelopes = {}
  for c in data :
    envelopes[c] = np.abs(hilbert(data[c]))
  return envelopes

"""# Graph Construction

preprocess function does the preprocessing of the data.

"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

class ButterworthFilter:
    def __init__(self, fs):
        self.fs = fs

    def bandpass(self, lowcut, highcut, order=5):
        return butter(order, [lowcut, highcut], fs=self.fs, btype='band')

    def bandpass_filter(self, data, lowcut, highcut, order=5):
        b, a = self.bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y

def preprocess (data) :
  filtered = {}
  filt = ButterworthFilter(256)
  for channel in data:
    filtered[channel] = filt.bandpass_filter(data[channel], 1, 70)
  envelopes = pd.DataFrame(compute_envelopes(filtered))
  return envelopes


import pickle

"""load the ids of the recordings ( they are not always sequential ) from the pickle file we have stored them at for a given subject."""

def get_recording_ids (subject_id, source_path) :
  with open(source_path+get_subject_directory(subject_id)+'/recording_ids.pickle', 'rb') as handle :
    recording_ids = pickle.load(handle)

  return recording_ids
 

"""set the interval for given data. Return the start times of each interval."""

import numpy as np
import matplotlib.pyplot as plt


def set_interval(data, interval_length_p, interval_step_p):

  # If interval_step_p is set to 'max',
  # we get a signle interval, taking up the whole timeseries
  if interval_step_p =='max' :
   interval_step = len(data.index)
  else :
    interval_step = interval_step_p

  interval_start = list(range(0, len(data.index), interval_step))
  return (interval_start, interval_step)

"""Calculate the correlations between the channels"""

def calculate_correlations(channels, data, interval_start, interval_length=1000):
  pairs = set()
  for channel0 in channels :
    for channel1 in channels :
      if channel1==channel0 : continue
      pairs.add(frozenset([channel0,channel1]))

  correlations = {}
  for interval in interval_start :
    correlations[interval]={}
    for pair in pairs :
     ch0, ch1 = pair
     correlations[interval][pair] = np.corrcoef(
                                      data[ch0][interval:interval+interval_length],
                                     data[ch1][interval:interval+interval_length]
                                   )[0][1]
      #print(ch0, ch1, ':', correlations[pair])
  return correlations

"""**FUNCTION TO GENERATE BRAIN GRAPH:**

create a matrix using the correlations caclulated which will be the brain graph.
"""

# Create a 2-D adjacency matrix from the previously calculated `correlations`
# for a specified interval
def generate_brain_graph(channels, correlations):
  matrix = []
  for c0 in channels :
    row = []

    for c1 in channels :

      # If same channel (diagonal) correlation is 1
      if c0==c1 :
          row.append(1)
      else :
         row.append(correlations[frozenset([c0,c1])])

    matrix.append(row)

  return matrix

"""Generate such Brain graphs for every interval using the correlations"""

def generate_brain_graphs (channels, correlations, interval_start, interval_length) :  
  matrices = []
  for interval in interval_start :
    matrix = generate_brain_graph(channels, correlations[interval])
    matrices.append(matrix)

  return matrices

"""extract a time series of the seizure in the form  0 for no seizure 1 for seizure creating a binary timeseries indicating the series."""

def extract_seizure_timeseries (data, seizures) :
  #seizure_ts = pd.DataFrame(0, index=data.index, columns=['seizure'])
  seizure_ts = pd.Series(0, index=data.index)

  for seizure in seizures :
    seizure_ts.loc[seizure[0]:seizure[1]] = 1

  return seizure_ts

"""get the seizure timeseries and extract the amount of seizure presence in each interval."""

def calculate_seizure_presence(interval_start, interval_length, seizure_ts):
  seizure_presence = []
  for interval in interval_start :
    window = seizure_ts[interval:interval+interval_length]
    presence = window.sum()/interval_length
    seizure_presence.append(presence)
  
  return seizure_presence

"""pair the x and y values to create a training set for the neural network"""

# x : the brain graph for each interval
# y : the seizure presence for each interval
# returns
def pair_x_y (x, y) :
  pairs = []
  for x_val, y_val in zip(x,y) :
    pairs.append([x_val, y_val])
  return pairs




"""#MORE THAN ONE RECORDING:

Do everything we did for one recording for all recordings of one patient.
"""

def load_patient (subject_p, target_path) :
  # Download file containing seizures of patient
  load_patient_seizures(subject_p, target_path)

  # Load the number of files to read
  #number_of_files = count_word_occurrences(get_patient_summary_path(subject_p), 'File Name')

  save_recording_ids(subject_p, target_path, target_path)
  recording_ids = get_recording_ids(subject_p, target_path)

  seizure_seconds = {}
  seizure_frames = {}

  print(recording_ids)

  #for i in range(1, number_of_files+1) :
  #  id = str(i).zfill(2)

  for id in recording_ids :
    print(f'Loading recording {id}\n')
    # Download the recording file
    load_edf_data(subject_p, id, target_path)

    '''
    if new_filename is None :
      number_of_files += 1
    '''


def save_recording_ids (subject_p, source_path, target_path) :
  summary_filename = get_patient_summary_path(subject_p)

  recording_ids = []
  with open(source_path+summary_filename, 'r', encoding='latin-1') as f:
    content = f.read()

    file_info = content.split("\n\n")
    for info in file_info:
      if 'File Name' in info :
        id = info.split('.')[0].split('_')[1]
        recording_ids.append(id)

    pathh = target_path+get_subject_directory(subject_p)+'/recording_ids.pickle'
    print(pathh)
    with open(target_path+get_subject_directory(subject_p)+'/recording_ids.pickle', 'wb') as handle :
      pickle.dump(recording_ids, handle)


def get_seizure_times_in_frames (seizure_times) : 
  return pd.Series([[256*seconds for seconds in s] for s in seizure_times])


def extract_patient_data (subject_p, source_path=base_path, target_path=base_path) :   
  source_subject_path = source_path+get_subject_directory(subject_p)+'/'
  target_subject_path = target_path+get_subject_directory(subject_p)

  if not (os.path.exists(target_subject_path)) :
    os.makedirs(target_subject_path)

  target_subject_path += '/'
  print(subject_p)
  save_recording_ids(subject_p, source_path, target_path)
  recording_ids = get_recording_ids(subject_p, target_path)

  #recordings = []
  seizure_seconds = {}
  seizure_frames = {}
  print(source_path)
  print(target_path)
  #for filename in os.listdir(source_subject_path):
  for rec_id in recording_ids :


    edf_filepath = source_path+get_data_path(subject_p, rec_id, 'edf')
    new_filepath = target_path+get_data_path(subject_p, rec_id, 'csv')
    if not os.path.isfile(new_filepath) :
      # Load the file into a dataframe
      data, channels = load_dataframe(edf_filepath)
      #rec_id = filename[-6:-4]
      data.to_csv(new_filepath, index=False)


    #if filename == 'seizures.edf' : continue
    #elif filename == 'recording_ids.pickle' : continue

    #f = os.path.join(source_subject_path, filename)
    #if not os.path.isfile(f):
    #  continue
    #print(source_subject_path+filename)


    # Save the recording into a dictionary
    #recordings[i] = data

    #seizure_seconds[rec_id] = pd.Series(extract_seizure_times(get_patient_summary_path(subject_p), 'chb'+subject_p+'_'+rec_id+'.edf')) # loading seizure times for current filename
    #print(rec_id)
    seizure_seconds[rec_id] = pd.Series(extract_seizure_times(subject_p, rec_id, source_path)) # loading seizure times for current filename
    #print(seizure_seconds[rec_id])
    seizure_frames[rec_id] = pd.Series([[256*seconds for seconds in s] for s in seizure_seconds[rec_id]])
    #print(seizure_frames[rec_id])
    #recordings.append(rec_id)

  print(target_subject_path+'seizure_seconds.pickle')
  with open(target_subject_path+'seizure_seconds.pickle', 'wb') as handle :
    pickle.dump(seizure_seconds, handle)

  with open(target_subject_path+'seizure_frames.pickle', 'wb') as handle :
    pickle.dump(seizure_frames, handle)




def get_subject_recording_x_y_data (subject_id, recording_id, source_path='/content/output/') :
  with open(source_path+get_subject_directory(subject_id)+f'/brain_graph_and_seizure_presence_pairs_{recording_id}.pickle', 'rb') as handle :
    paired_data = pickle.load(handle)
  return paired_data

# Returns a dictionary, with the recording id as the key,
# and the paired x,y data as the value
#
def get_subject_x_y_data (subject_id, source_path) :
  recording_ids = get_recording_ids(subject_id, source_path)
  print("recording_ids : ")
  print(recording_ids)
  all_paired_data = {}
  for id in recording_ids :
    all_paired_data[id] = get_subject_recording_x_y_data(subject_id, id, source_path)

  return all_paired_data

def process_patient (subject_id, interval_length_p = interval_length_p, interval_step_p = interval_step_p, source_path='/content/output/', target_path='/content/output/') :
  # Read the file with recording ids
  recording_ids = get_recording_ids(subject_id, source_path)

  interval_starts = {}
  interval_lengths = {}

  seizure_times_frames = load_pickle(source_path+get_seizure_frames_path(subject_id))

  seizure_ts_filename = target_path+get_subject_directory(subject_id)+f'/seizure_time_series.pickle'
  if os.path.isfile(seizure_ts_filename) :
    with open(seizure_ts_filename, 'rb') as handle :
      seizure_time_series = pickle.load(handle)
  else :
    seizure_time_series = {}

  print(seizure_times_frames)
  print(subject_id)
  print()
  print(list(seizure_times_frames.keys()))
  print(recording_ids)



  # For each recording,
  for id in recording_ids:
    if id in seizure_time_series : continue

    # Read dataframe
    recording_data = pd.read_csv(target_path+get_data_path (subject_id, id, 'csv'))
    channels = recording_data.columns.values.tolist()

    # Create envelopes and save them in target directory
    envelopes = preprocess(recording_data)


    envelopes.to_csv(target_path+get_subject_directory(subject_id)+f'/preprocessed_{id}.csv', index=False)

    # Create intervals
    interval_starts[id], interval_lengths[id] = set_interval(envelopes, interval_length_p, interval_step_p)

    # Create correlations
    correlations = calculate_correlations(channels, envelopes, interval_starts[id], interval_lengths[id])

    # Create brain graphs
    brain_graphs = generate_brain_graphs(channels, correlations, interval_starts[id], interval_lengths[id])

    '''
    with open(target_path+get_subject_directory(subject_id)+f'/brain_graph_{id}.pickle', 'wb') :
      pickle.dump(brain_graphs, handle)
    '''
    #seizure_times = extract_seizure_times(subject_id, id, source_path)
    #seizure_times_frames = get_seizure_times_in_frames(seizure_times)

    #seizure_times = load_pickle(get_seizure_seconds_path())

    # Get timeseries of seizure presence in each frame (timeseries with 0 or 1)
    '''
    with open(source_path+get_subject_directory(subject_id)+'/seizure_frames.pickle', 'rb') as handle :
      seizure_frames = pickle.load(handle)
    '''
    #print(seizure_times_frames[id])
    seizure_ts = extract_seizure_timeseries(envelopes, seizure_times_frames[id])
    seizure_time_series[id] = seizure_ts
    #print(seizure_time_series[id])
    #plt.plot(range(0,len(seizure_time_series[id])),seizure_time_series[id])
    #plt.show()

    print()
    print()
    print()

    # Get perc of seizure frames for each interval
    seizure_presence = calculate_seizure_presence(interval_starts[id], interval_lengths[id], seizure_ts)

    # Get paired data
    paired_data = pair_x_y(x=brain_graphs, y=seizure_presence)
    with open(target_path+get_subject_directory(subject_id)+f'/brain_graph_and_seizure_presence_pairs_{id}.pickle', 'wb') as handle :
      pickle.dump(paired_data, handle)


  # Save interval starts
  with open(target_path+get_subject_directory(subject_id)+f'/interval_starts.pickle', 'wb') as handle :
    pickle.dump(interval_starts, handle)

  # Save interval lengths
  with open(target_path+get_subject_directory(subject_id)+f'/interval_lengths.pickle', 'wb') as handle :
    pickle.dump(interval_lengths, handle)

  with open(target_path+get_subject_directory(subject_id)+f'/seizure_time_series.pickle', 'wb') as handle :
    pickle.dump(seizure_time_series, handle)


import pickle
#xy = get_subject_x_y_data (subject, processed_path)



def concat_recordings (subject) :
  
  with open(processed_path+get_subject_directory(subject)+f'/seizure_time_series.pickle', 'rb') as handle :
    seizure_timeseries = pickle.load(handle)
    
  xy = get_subject_x_y_data (subject, processed_path)
  flattened_seizure_timeseries = []
  flattened_xy_data = []
  lengths = []
  print(subject)
  for rec_id in xy :
    
    for s in xy[rec_id] :
      curshape = (len(s[0]), len(s[0][0]))
      if curshape != (23,23) :
        print('issue xy data of patient', subject, 'rec', rec_id)
        print(curshape)
        exit(1)
        
    lengths.append(len(xy[rec_id]))
    flattened_xy_data.extend(xy[rec_id])
    flattened_seizure_timeseries.extend(seizure_timeseries[rec_id])
    
  print( len(flattened_xy_data), len(flattened_xy_data[0]))
  print(np.sum(lengths))

  for s in flattened_xy_data :
    curshape = (len(s[0]), len(s[0][0]))
    if curshape!=(23,23) :
      print('issue in concat, subject', subject)
      
  x_data = np.array([xyi[0] for xyi in flattened_xy_data])
  y_data = np.array([[xyi[1]] for xyi in flattened_xy_data])

  return x_data, y_data, flattened_seizure_timeseries

import copy




   






"""Create the Convolutional Neural Network to train."""

import random
##all_indices = np.array(list(range(0, len(x_data))))

x_data = []
y_data = []
all_indices = list(range(0, len(x_data)))
flattened_seizure_timeseries = []





def split_shuffled (indices) :
  train_indices = random.sample(indices, int(0.8*len(indices)))
  test_indices = np.delete(indices, train_indices)
  
  return train_indices, test_indices



def split_rightmost (indices, per=0.2) :
  train_indices = indices[:int(per*(len(indices)))]
  test_indices = indices[int(per*(len(indices))):]

  return train_indices, test_indices


def split_equal_class_training (indices, y_data, per=1,threshold=0) :
  def shuffle_split (indices) :
    random.shuffle(indices)
    n_training = int(np.ceil(len(indices)*per))
    n_testing = len(indices) - n_training

    train_indices = indices[:n_training]
    test_indices = indices[n_training:]

    return train_indices, test_indices

  zeros = np.where(y_data<=threshold)[0]
  nonzeros = np.where(y_data>threshold)[0]



  # Split the indices of non-zero y values into training and testing
  train_nonzeros, test_nonzeros = shuffle_split(nonzeros)


  # Split the indices of zero y values into training and testing
  train_zeros, test_zeros       = shuffle_split(zeros)


  # Place unused training zeros to the testing set
  test_zeros                    = np.concatenate((test_zeros, train_zeros[len(train_nonzeros):]))
  # and limit the number of zeros in the training set to be equal
  # to the number of non-zero in the training set
  train_zeros                   = train_zeros[:len(train_nonzeros)]

  # Merge zero and non-zero training sets to create the final training set
  training = np.concatenate((train_zeros,train_nonzeros))
  #random.shuffle(training)


  # Same for testing set
  #testing = np.concatenate((test_zeros,test_nonzeros))

  #random.shuffle(testing)

  #return training, testing
  return train_zeros, train_nonzeros

