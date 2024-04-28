subject_ids = [str(i).zfill(2) for i in range(1, 24) if i not in [17,13,15,16,18,19,20,23]]
xnonz = []
xz = []
ynonz = []
yz = []


def load_equal_data(training_set_file = f'all_patients_all_recordings_training_set.pickle')

  if os.path.isfile(training_set_file) :
    x_data, y_data = pickle.load(open(training_set_file, 'rb'))

  else : 
  
    lengths = []
  
    #flattened_seizure_timeseries = []
    for sid in subject_ids[:] :
        x_data_s, y_data_s, flattened_seizure_timeseries_s = concat_recordings(sid)
      
        #train_indices, test_indices = split_equal_class_training(range(len(x_data_s)), y_data_s)
        a, b = split_equal_class_training(range(len(x_data_s)), y_data_s)
        #a, b = split_rightmost(range(len(x_data_s)), 0.5)
        train_indices = list(a)+list(b)
        xz.extend(x_data_s[a])
        yz.extend(y_data_s[a])
        xnonz.extend(x_data_s[b])
        ynonz.extend(y_data_s[b])
        lengths.append(len(train_indices))
        for s in x_data_s[train_indices] :
          curshape = (len(s), len(s[0]))
          if curshape != (23,23) :
            print('f', sid)
            print(curshape)
            exit(1)
      
      #x_data.extend(x_data_s[train_indices])
      #y_data.extend(y_data_s[train_indices])
      #flattened_seizure_timeseries.extend(flattened_seizure_timeseries_s[train_indices])

  for s in x_data :
    curshape = (len(s), len(s[0]))
    if curshape != (23,23) :
      print('issue in concatenated subjects')
      print(curshape)
      exit(1)

  x_data = list(xnonz)+list(xz)
  y_data = list(ynonz)+list(yz)
  pickle.dump((x_data, y_data), open(training_set_file,'wb'))
  


  print('original data length', len(x_data))


  adj_matrix_data = np.array(x_data)#[train_indices]
  x_data = adj_matrix_data
  labels_data = np.array(y_data)#[train_indices]

#test_adj_matrix_data = x_data[test_indices]
#validation_adj_matrix_data = np.delete(validation_adj_matrix_data, test_indices)

#test_labels_data = y_data[test_indices]
#validation_labels_data = np.delete(validation_labels_data, test_indices)

#plt.figure()
#plt.plot(range(len(y_data)), y_data)
#plt.show()

#plt.figure()
#plt.plot(range(len(y_data)), y_data)
#plt.show()

"""For interpatient testing:"""

#p6x, p6y = concat_recordings('06')
#p1x, p1y = concat_recordings('01')
#
#adj_matrix_data = p1x
#labels_data = p1y
#
#test_adj_matrix_data = p6x
#test_labels_data = p6y

"""CNN training:"""

import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D


#input_shape = (23, 23, 1
#adj_matrix_data = np.stack(adj_matrix_data)
print(len(adj_matrix_data))
print(len(labels_data))
#print(len(test_adj_matrix_data))
#print(len(test_labels_data))
print(adj_matrix_data.shape)
print(adj_matrix_data[0].shape)



input_shape = (*adj_matrix_data[0].shape, 1)
#change labels data to be categorical so 0.0 to 0.1 is category 0 so neuron 0 is activated
#labels_data = tf.keras.utils.to_categorical(labels_data, num_classes=10)
#test_labels_data = tf.keras.utils.to_categorical(test_labels_data, num_classes=10)

print(input_shape)

#validation_adj_matrix_data = np.expand_dims(validation_adj_matrix_data, axis=0)  # Add batch dimension
'''
for i in test_indices :
  print(i)
  del validation_adj_matrix_data[i]
  del validation_labels_data[i]
'''


def get_model():
  model_path = 'GD3iz/current_model_5.keras'
  if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
  else:
  # Define the model using tf.keras.Sequential
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
      tf.keras.layers.Dropout(rate=0.4),
      tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dropout(rate=0.3),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dropout(rate=0.3),
      tf.keras.layers.Dense(units=16, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
  ]) 

    # Compile the model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=[tf.keras.metrics.MeanSquaredLogarithmicError(), "accuracy"])

    return model

  
# plot the input of the model to see if it is correct
#model.summary()
#plot adj_matrix_data
    
import matplotlib.pyplot as plt

# Plot adj_matrix_data
#for i in range(4):
#    plt.figure()
#    plt.imshow(adj_matrix_data[i][:, :], cmap='gray')
#    plt.title(f"Adjacency Matrix {i+1}")
#    plt.show()

from sklearn.utils.class_weight import compute_class_weight

#flattened_labels = labels_data.flatten()
#class_weights = compute_class_weight(class_weight={i: 0.001 if i<0.1 else 0.9 for i in np.unique(flattened_labels)},classes= np.unique(flattened_labels),y= flattened_labels)
# Convert class weights to dictionary format
#class_weight_dict = {i: weight for i, weight in enumerate(class_weights)} 

# Train the model
def train_model(model):
  model.fit(adj_matrix_data, labels_data, epochs=200, batch_size=500)



# Save the updated model
#model.save(model_path)



"""Calculate accuracy"""

# Classify a new graph

correct = 0
#new_adj_matrix = np.random.randint(2, size=(1, 27, 27))


#for new_adj_matrix, new_label in zip(validation_adj_matrix_data, validation_labels_data):
#    new_adj_matrix = np.expand_dims(new_adj_matrix, axis=0)  # Add batch dimension
#    predicted_labels = model.predict(new_adj_matrix)
#    predicted_class = predicted_labels#np.round(predicted_labels).astype(int)
#    #print()
#    #print('Predicted Class:', predicted_class)
#    #print('True Class:', new_label)
#    if predicted_labels < new_label+0.2 and predicted_labels > new_label-0.2:
#      correct = correct + 1
#
#print(correct)
#print(len(test_adj_matrix_data))
#model_prediction_output = model.predict(test_adj_matrix_data)
#predicted_labels = [label for entry in model_prediction_output for label in entry]
#test_labels = [label for entry in test_labels_data for label in entry]
#
#
#
#labels = pd.DataFrame({'predicted':predicted_labels, 'true':test_labels})
#
#labels['residuals'] = labels['predicted']-labels['true']
#accuracy = len(labels[abs(labels['residuals'])<0.005].index) / len(labels.index)
#
#print('Mean Error:', labels['residuals'].mean(axis=0))
#print('Abs Mean Error:', labels['residuals'].abs().mean(axis=0))
#print('Accuracy  :', accuracy)
#
#print(labels)

"""# Plots
Plots to examine the validity of our calculations
"""

def plot_results(x_data, model):
  print(len(x_data))
  model_prediction_output = model.predict(x_data)
  predicted_labels = [label for entry in model_prediction_output for label in entry]

  #all_frames = list(range(0, len(flattened_seizure_timeseries)))
  window_frames = list(range(0, len(predicted_labels)))

  #print(all_frames[:10])
  #print(window_frames[:10])



  plt.figure()
  #plt.plot(window_frames, seizure_presence)
  #plot presence
  plt.plot(window_frames, y_data)
  plt.show()
  plt.figure()
  plt.plot(window_frames, y_data)
  #plt.show()
  #plt.figure()
  # plot predictions
  plt.plot(window_frames, predicted_labels, 'r')
  plt.show()
  plt.figure()
  #plot thresholded predictions
  #threshold = 0.005
  #thresholded_predictions = [0 if v<threshold else 1 for v in predicted_labels]
  #plt.plot(window_frames, thresholded_predictions)
  #plt.show()
  #plt.figure()
