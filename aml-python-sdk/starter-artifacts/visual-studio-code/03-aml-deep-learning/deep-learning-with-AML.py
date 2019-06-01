'''
Train a deep learning model
In this notebook you will train a deep learning model to classify the descriptions of car components as compliant or non-compliant.

Each document in the supplied training data set is a short text description of the component as documented by an authorized technician. The contents include:

Manufacture year of the component (e.g. 1985, 2010)
Condition of the component (poor, fair, good, new)
Materials used in the component (plastic, carbon fiber, steel, iron)
The compliance regulations dictate: Any component manufactured before 1995 or in fair or poor condition or made with plastic or iron is out of compliance.

For example:

Manufactured in 1985 made of steel in fair condition -> Non-compliant
Good condition carbon fiber component manufactured in 2010 -> Compliant
Steel component manufactured in 1995 in fair condition -> Non-Compliant
The labels present in this data are 0 for compliant, 1 for non-compliant.

The challenge with classifying text data is that deep learning models only undertand vectors (e.g., arrays of numbers) and not text. To encode the car component descriptions as vectors, we use an algorithm from Stanford called GloVe (Global Vectors for Word Representation). GloVe provides us pre-trained vectors that we can use to convert a string of text into a vector.

Setup
To begin, you will need to provide the following information about your Azure Subscription.

In the following cell, be sure to set the values for subscription_id, resource_group, workspace_name and workspace_region as directed by the comments (these values can be acquired from the Azure Portal).

If you are in doubt of any of these values, do the following:

Navigate to the Azure Portal and login with the credentials provided.
From the left hand menu, under Favorites, select Resource Groups.
In the list, select the resource group with the name similar to TBD-XXXXX.
From the Overview tab, capture the desired values.
Execute the following cell by selecting the >|Run button in the command bar above.
'''

#%%
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "xxx-xxx-xxx"

#Provide values for the existing Resource Group 
resource_group = "Deep-Learning-Lablet"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "deep-learning-workspace"
workspace_region = "eastus"

experiment_name = 'deep-learning'
project_folder = './dl'
deployment_folder = './deploy'

# this is the URL to the CSV file containing the GloVe vectors
glove_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/glove.6B.100d.txt"

# this is the URL to the CSV file containing the care component descriptions
data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/connected-car_components.csv"

# this is the name of the AML Compute cluster
cluster_name = "gpucluster"

'''
Create the Azure Machine Learning resources
The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:

Creating a Workspace that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
Creating Experiments in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
Creating Compute resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU.
Using Automated Machine Learning (AutoML) to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
Packaging a Docker Image that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the Web Service.
In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following cell to do so:
'''

#%%
import logging
import os
import random
import re

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.model import Model
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core import Workspace

'''
Create and connect to an Azure Machine Learning Workspace
Run the following cell to create a new Azure Machine Learning Workspace and save the configuration to disk (next to the Jupyter notebook).

Important Note: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read Workspace configuration succeeded.
'''

#%%
# By using the exist_ok param, if the worskpace already exists you get a reference to the existing workspace
# allowing you to re-run this cell multiple times as desired (which is fairly common in notebooks).
ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

ws.write_config()
print('Workspace configuration succeeded')

'''
Create AML Compute Cluster
Now you are ready to create the GPU compute cluster. Run the following cell to create a new compute cluster (or retrieve the existing cluster if it already exists). The code below will create a GPU based cluster where each node in the cluster is of the size Standard_NC12, and the cluster is restricted to use 1 node. This will take couple of minutes to create.
'''

#%%
### Create AML CPU based Compute Cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC12',
                                                           min_nodes=1, max_nodes=1)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use the 'status' property to get a detailed status for the current AmlCompute. 
print(compute_target.status.serialize())

'''
# Create the Keras estimator
'''

#%%
from azureml.train.dnn import TensorFlow

keras_est = TensorFlow(source_directory=project_folder,
                       compute_target=compute_target,
                       entry_script='train.py',
                       conda_packages=['pandas'],
                       pip_packages=['keras==2.2.4'], # just add keras through pip
                       use_gpu=True)

'''
Remotely train a deep learning model using the Azure ML Compute

In the following cells, you will not train the model against the data you just downloaded using the resources provided by Azure Notebooks. Instead, you will deploy an Azure ML Compute cluster that will download the data and use a trainings script to train the model. In other words, all of the training will be performed remotely with respect to this notebook.
'''

# create project folder
#%%
if not os.path.exists(project_folder):
    os.makedirs(project_folder)

# Create the training script

#%%
%%writefile $project_folder/train.py

import os
import numpy as np
import pandas as pd

import keras
from keras import models 
from keras import layers
from keras import optimizers

def get_data():
    data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/connected-car_components.csv"
    car_components_df = pd.read_csv(data_url)
    components = car_components_df["text"].tolist()
    labels = car_components_df["label"].tolist()
    return { "components" : components, "labels" : labels }

def download_glove():
    print("Downloading GloVe embeddings...")
    import urllib.request
    urllib.request.urlretrieve('https://databricksdemostore.blob.core.windows.net/data/connected-car/glove.6B.100d.txt', 'glove.6B.100d.txt')
    print("Download complete.")

download_glove()

# Load the car components labeled data
print("Loading car components data...")
data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/connected-car_components.csv"
car_components_df = pd.read_csv(data_url)
components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()
print("Loading car components data completed.")

# split data 60% for trianing, 20% for validation, 20% for test
print("Splitting data...")
train, validate, test = np.split(car_components_df.sample(frac=1), [int(.6*len(car_components_df)), int(.8*len(car_components_df))])
print(train.shape)
print(test.shape)
print(validate.shape)

# use the Tokenizer from Keras to "learn" a vocabulary from the entire car components text
print("Tokenizing data...")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100                                           
training_samples = 90000                                 
validation_samples = 5000    
max_words = 10000      

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])                     
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

x_test = data[training_samples + validation_samples:]
y_test = labels[training_samples + validation_samples:]
print("Tokenizing data complete.")

# apply the vectors provided by GloVe to create a word embedding matrix
print("Applying GloVe vectors...")
glove_dir =  './'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
print("Applying GloVe vectors compelted.")

# use Keras to define the structure of the deep neural network   
print("Creating model structure...")
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# fix the weights for the first layer to those provided by the embedding matrix
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
print("Creating model structure completed.")

print("Training model...")
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=1, #limit the demo to 1 epoch
                    batch_size=32,
                    validation_data=(x_val, y_val))
print("Training model completed.")

print("Saving model files...")
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
# save model
model.save('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
print("Saving model files completed.")

'''
Submit the training run
The code pattern to submit a training run to Azure Machine Learning compute targets is always:

Create an experiment to run.
Submit the experiment.
Wait for the run to complete.
'''

# Create the experiment
#%%
experiment = Experiment(ws, experiment_name)

# Submit the experiment
#%%
run = experiment.submit(keras_est)

'''
Wait for the run to complete by executing the following cell. Note that this process will perform the following:

Build and deploy the container to Azure Machine Learning compute (~8 minutes)
Execute the training script (~2 minutes)
If you change only the training script and re-submit, it will run faster the second time because the necessary container is already prepared so the time requried is just that for executing the training script.
'''

#%%
run.wait_for_completion(show_output = True)

'''
Download the model files from the run
In the training script, the Keras model is saved into two files, model.json and model.h5, in the outputs/models folder on the GPU cluster AmlCompute node. Azure ML automatically uploaded anything written in the ./outputs folder into run history file store. Subsequently, we can use the run object to download the model files. They are under the the outputs/model folder in the run history file store, and are downloaded into a local folder named model.
'''

#%%
# create a model folder in the current directory
os.makedirs('./model', exist_ok=True)

for f in run.get_file_names():
    if f.startswith('outputs/model'):
        output_file_path = os.path.join('./model', f.split('/')[-1])
        print('Downloading from {} to {} ...'.format(f, output_file_path))
        run.download_file(name=f, output_file_path=output_file_path)

'''
Restore the model from model.h5 file
'''

#%%
import tensorflow as tf

model = tf.keras.models.load_model('./model/model.h5')
print("Model loaded from disk.")
print(model.summary())

'''
Evaluate the model on test data
You can also evaluate how accurately the model performs against data it has not seen. Run the following cell to load the test data that was not used in either training or evaluating the model.
'''

#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the car components labeled data
car_components_df = pd.read_csv(data_url)
components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()

maxlen = 100                                           
training_samples = 90000                                 
validation_samples = 5000    
max_words = 10000      

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])                     
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_test = data[training_samples + validation_samples:]
y_test = labels[training_samples + validation_samples:]

'''
Run the following cell to see the accuracy on the test set (it is the second number in the array displayed, on a scale from 0 to 1).
'''

#%%
print('model.evaluate will print the following metrics: ', model.metrics_names)
print(model.evaluate(x_test, y_test))
