# # Train a deep learning model
# In this notebook you will train a deep learning model to classify the descriptions of car components as compliant or non-compliant. 
# 
# Each document in the supplied training data set is a short text description of the component as documented by an authorized technician. 
# The contents include:
# - Manufacture year of the component (e.g. 1985, 2010)
# - Condition of the component (poor, fair, good, new)
# - Materials used in the component (plastic, carbon fiber, steel, iron)
# 
# The compliance regulations dictate:
# *Any component manufactured before 1995 or in fair or poor condition or made with plastic or iron is out of compliance.*
# 
# For example:
# * Manufactured in 1985 made of steel in fair condition -> **Non-compliant**
# * Good condition carbon fiber component manufactured in 2010 -> **Compliant**
# * Steel component manufactured in 1995 in fair condition -> **Non-Compliant**
# 
# The labels present in this data are 0 for compliant, 1 for non-compliant.
# 
# The challenge with classifying text data is that deep learning models only undertand vectors (e.g., arrays of numbers) and not text. To encode the car component descriptions as vectors, we use an algorithm from Stanford called [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/). GloVe provides us pre-trained vectors that we can use to convert a string of text into a vector.

# ### Setup
# To begin, you will need to provide the following information about your Azure Subscription.
# 
# **If you are using your own Azure subscription, please provide names for subscription_id, resource_group, workspace_name and workspace_region to use.** Note that the workspace needs to be of type [Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace).
# 
# **If an environment is provided to you be sure to replace XXXXX in the values below with your unique identifier.**
# 
# In the following cell, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments (*these values can be acquired from the Azure Portal*).
# 
# To get these values, do the following:
# 1. Navigate to the Azure Portal and login with the credentials provided.
# 2. From the left hand menu, under Favorites, select `Resource Groups`.
# 3. In the list, select the resource group with the name similar to `XXXXX`.
# 4. From the Overview tab, capture the desired values.
# 
# Execute the following cell by selecting the `>|Run` button in the command bar above.

# In[ ]:

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "" # <- needs to be the subscription with the Quick-Starts resource group

#Provide values for the existing Resource Group 
resource_group = "Quick-Starts-XXXXX" # <- replace XXXXX with your unique identifier

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "quick-starts-ws-XXXXX" # <- replace XXXXX with your unique identifier
workspace_region = "eastus" # <- region of your Quick-Starts resource group

# In[ ]:

experiment_name = 'deep-learning'
project_folder = './dl'
deployment_folder = './deploy'

# this is the URL to the CSV file containing the GloVe vectors
glove_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
             'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
             'quickstarts/connected-car-data/glove.6B.100d.txt')

# this is the URL to the CSV file containing the care component descriptions
data_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
            'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
            'quickstarts/connected-car-data/connected-car_components.csv')

# this is the name of the AML Compute cluster
cluster_name = "gpucluster"


# # Create the Azure Machine Learning resources

# The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:
# - Creating a **Workspace** that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
# - Creating **Experiments** in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
# - Creating **Compute** resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU. 
# - Using **Automated Machine Learning (AutoML)** to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
# - Packaging a Docker **Image** that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
# - Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the **Web Service**.
# 
# In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following cell to do so:

# In[ ]:

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


# ## Create and connect to an Azure Machine Learning Workspace
# Run the following cell to create a new Azure Machine Learning **Workspace** and save the configuration to disk (next to the Jupyter notebook). 
# 
# **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.

# In[ ]:

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


# ### Create AML Compute Cluster
# Now you are ready to create the GPU compute cluster. Run the following cell to create a new compute cluster (or retrieve the existing cluster if it already exists). The code below will create a *GPU based* cluster where each node in the cluster is of the size `Standard_NC12`, and the cluster is restricted to use 1 node. This will take couple of minutes to create.

# In[ ]:

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


# ### Create the Keras Estimator

# In[ ]:

from azureml.train.dnn import TensorFlow

keras_est = TensorFlow(source_directory=project_folder,
                       compute_target=compute_target,
                       entry_script='train.py',
                       conda_packages=['pandas'],
                       pip_packages=['keras==2.2.4'], # just add keras through pip
                       use_gpu=True)


# ## Remotely train a deep learning model using the Azure ML Compute
# In the following cells, you will *not* train the model against the data you just downloaded using the resources provided by Azure Notebooks. Instead, you will deploy an Azure ML Compute cluster that will download the data and use a trainings script to train the model. In other words, all of the training will be performed remotely with respect to this notebook. 
# 

# In[ ]:

# create project folder
if not os.path.exists(project_folder):
    os.makedirs(project_folder)


# ### Create the training script

# In[ ]:

get_ipython().run_cell_magic('writefile', '$project_folder/train.py', '\nimport os\nimport numpy as np\nimport pandas as pd\n\nimport keras\nfrom keras import models \nfrom keras import layers\nfrom keras import optimizers\n\ndef get_data():\n    data_url = (\'https://quickstartsws9073123377.blob.core.windows.net/\'\n                \'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/\'\n                \'quickstarts/connected-car-data/connected-car_components.csv\')\n    car_components_df = pd.read_csv(data_url)\n    components = car_components_df["text"].tolist()\n    labels = car_components_df["label"].tolist()\n    return { "components" : components, "labels" : labels }\n\ndef download_glove():\n    print("Downloading GloVe embeddings...")\n    import urllib.request\n    glove_url = (\'https://quickstartsws9073123377.blob.core.windows.net/\'\n                 \'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/\'\n                 \'quickstarts/connected-car-data/glove.6B.100d.txt\')\n    urllib.request.urlretrieve(glove_url, \'glove.6B.100d.txt\')\n    print("Download complete.")\n\ndownload_glove()\n\n# Load the car components labeled data\nprint("Loading car components data...")\ndata_url = (\'https://quickstartsws9073123377.blob.core.windows.net/\'\n            \'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/\'\n            \'quickstarts/connected-car-data/connected-car_components.csv\')\ncar_components_df = pd.read_csv(data_url)\ncomponents = car_components_df["text"].tolist()\nlabels = car_components_df["label"].tolist()\nprint("Loading car components data completed.")\n\n# split data 60% for trianing, 20% for validation, 20% for test\nprint("Splitting data...")\ntrain, validate, test = np.split(car_components_df.sample(frac=1), [int(.6*len(car_components_df)), int(.8*len(car_components_df))])\nprint(train.shape)\nprint(test.shape)\nprint(validate.shape)\n\n# use the Tokenizer from Keras to "learn" a vocabulary from the entire car components text\nprint("Tokenizing data...")\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.preprocessing.sequence import pad_sequences\nimport numpy as np\n\nmaxlen = 100                                           \ntraining_samples = 90000                                 \nvalidation_samples = 5000    \nmax_words = 10000      \n\ntokenizer = Tokenizer(num_words=max_words)\ntokenizer.fit_on_texts(components)\nsequences = tokenizer.texts_to_sequences(components)\n\nword_index = tokenizer.word_index\nprint(\'Found %s unique tokens.\' % len(word_index))\n\ndata = pad_sequences(sequences, maxlen=maxlen)\n\nlabels = np.asarray(labels)\nprint(\'Shape of data tensor:\', data.shape)\nprint(\'Shape of label tensor:\', labels.shape)\n\nindices = np.arange(data.shape[0])                     \nnp.random.shuffle(indices)\ndata = data[indices]\nlabels = labels[indices]\n\nx_train = data[:training_samples]\ny_train = labels[:training_samples]\n\nx_val = data[training_samples: training_samples + validation_samples]\ny_val = labels[training_samples: training_samples + validation_samples]\n\nx_test = data[training_samples + validation_samples:]\ny_test = labels[training_samples + validation_samples:]\nprint("Tokenizing data complete.")\n\n# apply the vectors provided by GloVe to create a word embedding matrix\nprint("Applying GloVe vectors...")\nglove_dir =  \'./\'\n\nembeddings_index = {}\nf = open(os.path.join(glove_dir, \'glove.6B.100d.txt\'))\nfor line in f:\n    values = line.split()\n    word = values[0]\n    coefs = np.asarray(values[1:], dtype=\'float32\')\n    embeddings_index[word] = coefs\nf.close()\n\nprint(\'Found %s word vectors.\' % len(embeddings_index))\n\nembedding_dim = 100\n\nembedding_matrix = np.zeros((max_words, embedding_dim))\nfor word, i in word_index.items():\n    if i < max_words:\n        embedding_vector = embeddings_index.get(word)\n        if embedding_vector is not None:\n            embedding_matrix[i] = embedding_vector    \nprint("Applying GloVe vectors compelted.")\n\n# use Keras to define the structure of the deep neural network   \nprint("Creating model structure...")\nfrom keras.models import Sequential\nfrom keras.layers import Embedding, Flatten, Dense\n\nmodel = Sequential()\nmodel.add(Embedding(max_words, embedding_dim, input_length=maxlen))\nmodel.add(Flatten())\nmodel.add(Dense(32, activation=\'relu\'))\nmodel.add(Dense(1, activation=\'sigmoid\'))\nmodel.summary()\n\n# fix the weights for the first layer to those provided by the embedding matrix\nmodel.layers[0].set_weights([embedding_matrix])\nmodel.layers[0].trainable = False\nprint("Creating model structure completed.")\n\nprint("Training model...")\nmodel.compile(optimizer=\'rmsprop\',\n              loss=\'binary_crossentropy\',\n              metrics=[\'acc\'])\nhistory = model.fit(x_train, y_train,\n                    epochs=1, #limit the demo to 1 epoch\n                    batch_size=32,\n                    validation_data=(x_val, y_val))\nprint("Training model completed.")\n\nprint("Saving model files...")\n# create a ./outputs/model folder in the compute target\n# files saved in the "./outputs" folder are automatically uploaded into run history\nos.makedirs(\'./outputs/model\', exist_ok=True)\n# save model\nmodel.save(\'./outputs/model/model.h5\')\nprint("model saved in ./outputs/model folder")\nprint("Saving model files completed.")')


# ## Submit the training run

# The code pattern to submit a training run to Azure Machine Learning compute targets is always:
# 
# - Create an experiment to run.
# - Submit the experiment.
# - Wait for the run to complete.

# ### Create the experiment

# In[ ]:

experiment = Experiment(ws, experiment_name)


# ### Submit the experiment

# In[ ]:

run = experiment.submit(keras_est)


# Wait for the run to complete by executing the following cell. Note that this process will perform the following:
# - Build and deploy the container to Azure Machine Learning compute (~8 minutes)
# - Execute the training script (~2 minutes)
# 
# If you change only the training script and re-submit, it will run faster the second time because the necessary container is already prepared so the time requried is just that for executing the training script.

# In[ ]:

run.wait_for_completion(show_output = True)


# ## Download the model files from the run

# In the training script, the Keras model is saved into two files, model.json and model.h5, in the outputs/models folder on the GPU cluster AmlCompute node. Azure ML automatically uploaded anything written in the ./outputs folder into run history file store. Subsequently, we can use the run object to download the model files. They are under the the outputs/model folder in the run history file store, and are downloaded into a local folder named model.

# In[ ]:

# create a model folder in the current directory
os.makedirs('./model', exist_ok=True)

for f in run.get_file_names():
    if f.startswith('outputs/model'):
        output_file_path = os.path.join('./model', f.split('/')[-1])
        print('Downloading from {} to {} ...'.format(f, output_file_path))
        run.download_file(name=f, output_file_path=output_file_path)


# ## Restore the model from model.h5 file

# In[ ]:

from keras.models import load_model

model = load_model('./model/model.h5')
print("Model loaded from disk.")
print(model.summary())


# ## Evaluate the model on test data

# You can also evaluate how accurately the model performs against data it has not seen. Run the following cell to load the test data that was not used in either training or evaluating the model. 

# In[ ]:

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


# Run the following cell to see the accuracy on the test set (it is the second number in the array displayed, on a scale from 0 to 1).

# In[ ]:

print('model.evaluate will print the following metrics: ', model.metrics_names)
print(model.evaluate(x_test, y_test))
