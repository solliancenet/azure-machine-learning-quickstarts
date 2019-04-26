'''
Train a deep learning model
In this notebook you will train a deep learning model to classify the descriptions of car components as compliant or non-compliant. 
 
Each document in the supplied training data set is a short text description of the component as documented by an authorized technician. 
The contents include:
- Manufacture year of the component (e.g. 1985, 2010)
- Condition of the component (poor, fair, good, new)
- Materials used in the component (plastic, carbon fiber, steel, iron)

The compliance regulations dictate:
*Any component manufactured before 1995 or in fair or poor condition or made with plastic or iron is out of compliance.*

For example:
* Manufactured in 1985 made of steel in fair condition -> **Non-compliant**
* Good condition carbon fiber component manufactured in 2010 -> **Compliant**
* Steel component manufactured in 1995 in fair condition -> **Non-Compliant**

The labels present in this data are 0 for compliant, 1 for non-compliant.

The challenge with classifying text data is that deep learning models only undertand vectors (e.g., arrays of numbers) and not text. To encode the car component descriptions as vectors, we use an algorithm from Stanford called [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/). GloVe provides us pre-trained vectors that we can use to convert a string of text into a vector. 

Setup
To begin, you will need to provide the following information about your Azure Subscription. 

In the following cell, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments (*these values can be acquired from the Azure Portal*). Also provide the values for the pre-created AKS cluster (`aks_resource_group_name` and `aks_cluster_name`). 

If you are in doubt of any of these values, do the following:
1. Navigate to the Azure Portal and login with the credentials provided.
2. From the left hand menu, under Favorites, select `Resource Groups`.
3. In the list, select the resource group with the name similar to `tech-immersion-onnx-XXXXX`.
4. From the Overview tab, capture the desired values.

Execute the following cell by selecting the `>|Run` button in the command bar above.

Create the Azure Machine Learning resources

The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:
- Creating a **Workspace** that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
- Creating **Experiments** in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
- Creating **Compute** resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU. 
- Using **Automated Machine Learning (AutoML)** to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify. 
- Packaging a Docker **Image** that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
- Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the **Web Service**.

In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following cell to do so:
'''

# In[17]:
import numpy as np
import pandas as pd
import os
import json
import azureml
from azureml.core import Workspace
from azureml.core.model import Model
import keras
from keras import models
from keras.models import model_from_json
from keras import layers
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "xxx-xxx-xxx"

#Provide values for the existing Resource Group 
resource_group = "Deep-Learning-Lablet"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "deep-learning-workspace"
workspace_region = "eastus"

deployment_folder = './deploy'
onnx_export_folder = './onnx'

# this is the URL to the CSV file containing the GloVe vectors
glove_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/glove.6B.100d.txt"

# this is the URL to the CSV file containing the care component descriptions
data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/connected-car_components.csv"

# Load the car components labeled data
# In[20]:
car_components_df = pd.read_csv(data_url)
components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()

# Split the data into train and test
# In[21]:
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

'''
Restore the model from model.h5 file

The Keras model is saved in model.h5 file. Load a previously trained Keras model from the local **model** directory and review the model summary.
'''

# In[22]:
cwd = os.getcwd()
if cwd.endswith('/deploy'):
    os.chdir('../')

embedding_dim = 100
maxlen = 100                                             
max_words = 10000    

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# load weights into new model
model.load_weights("model/model.h5")
print("Model loaded from disk.")
print(model.summary())

'''
Converting a Keras model to ONNX
In the steps that follow, you will convert Keras model you just trained to the ONNX format. This will enable you to use this model for classification in a very broad range of environments, outside of Azure Databricks including:

- Web services 
- iOS and Android mobile apps
- Windows apps
- IoT devices

Converting a Keras model requires the use of the `winmltools`, `onnx` and `protobuf` libraries. Run the following cell to install these.
'''

# In[23]:
get_ipython().run_cell_magic('sh', '', 'pip install protobuf\npip install onnx\npip install winmltools')


# Convert the model to ONNX by running the following cell.

# In[24]:
import winmltools

# create a deployment folder in the current directory
os.makedirs(deployment_folder, exist_ok=True)

# create an onnx subfolder under deployment
os.makedirs(os.path.join(deployment_folder, onnx_export_folder), exist_ok=True)

onnx_model_name = 'component_compliance.onnx'
converted_model = winmltools.convert_keras(model, target_opset=7)
winmltools.save_model(converted_model, 
                      os.path.join(os.path.join(deployment_folder, onnx_export_folder), onnx_model_name))

'''
The above cell created a new file called `component_compliance.onnx` that contains the ONNX version of the model. 

To be able to use an ONNX model for inferencing, your environment only needs to have the `onnxruntime` installed. Run the following cell to install the `onnxruntime`. 
'''

# In[25]:
get_ipython().run_cell_magic('sh', '', 'pip install onnxruntime')

'''
Now try using this ONNX model to classify a component description by running the following cell. Remeber the prediction will be a value close to 0 (non-compliant) or to 1 (compliant).

Compare ONNX Inference Performace with Keras 
Create an onnxruntime InferenceSession and observe the expected input shape for inference. Classify a sample data from test set using both ONNX and Keras. Remeber the prediction will be a value close to 0 (non-compliant) or to 1 (compliant).

Next, we will evaluate the performance of ONNX and Keras by running the same sample 10,000 times. Make your observations comparing the performance of ONNX vs Keras for making inferences.
'''

# In[26]:
import onnxruntime
# Load the ONNX model and observe the expected input shape
onnx_session = onnxruntime.InferenceSession(
    os.path.join(os.path.join(deployment_folder, onnx_export_folder), onnx_model_name))
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name
print('Expected input shape: ', onnx_session.get_inputs()[0].shape)


# In[27]:
# Grab one sample from the test data set
x_test_float = np.reshape(x_test[1502].astype(np.float32), (1,100))
# Confirm that the input shape is same as expected input shape
print('Input shape: ', x_test_float.shape)

# Run an ONNX session to classify the sample.
print('ONNX prediction: ', onnx_session.run([output_name], {input_name : x_test_float}))

# Use Keras to make predictions on the same sample
print('Keras prediction: ', model.predict(x_test_float))

# Next we will compare the performance of ONNX vs Keras
import timeit
n = 10000


# In[28]:
start_time = timeit.default_timer()
for i in range(n):
    model.predict(x_test_float)
keras_elapsed = timeit.default_timer() - start_time
print('Keras performance: ', keras_elapsed)


# In[29]:
start_time = timeit.default_timer()
for i in range(n):
    onnx_session.run([output_name], {input_name : x_test_float})
onnx_elapsed = timeit.default_timer() - start_time
print('ONNX performance: ', onnx_elapsed)
print('ONNX is about {} times faster than Keras'.format(round(keras_elapsed/onnx_elapsed)))

'''
Deploy Deep Learning ONNX format model as a web service
To demonstrate one example of using the ONNX format model in a new environment, you will deploy the ONNX model to a webservice. On the web server, the only component required by the model is the ONNX Runtime, which is used to load the model and use it for scoring. Neither Keras nor TensorFlow are required on the web server.

In this case, you will use the Azure Machine Learning service SDK to programmatically create a Workspace, register your model, create a container image for the web service that uses it and deploy that image on to an Azure Container Instance.

Run the following cells to create some helper functions that you will use for deployment.
'''

# In[31]:
def getOrCreateWorkspace(subscription_id, resource_group, workspace_name, workspace_region):
    # By using the exist_ok param, if the workspace already exists we get a reference to the existing workspace instead of an error
    ws = Workspace.create(
        name = workspace_name,
        subscription_id = subscription_id,
        resource_group = resource_group, 
        location = workspace_region,
        exist_ok = True)
    return ws


# In[32]:
def deployModelAsWebService(ws, model_folder_path="models", model_name="component_compliance", 
                scoring_script_filename="scoring-service.py", 
                conda_packages=['numpy','pandas'],
                pip_packages=['azureml-sdk','onnxruntime'],
                conda_file="dependencies.yml", runtime="python",
                cpu_cores=1, memory_gb=1, tags={'name':'scoring'},
                description='Compliance classification web service.',
                service_name = "complianceservice"
               ):
    # notice for the model_path, we supply the name of the outputs folder without a trailing slash
    # this will ensure both the model and the customestimators get uploaded.
    print("Registering and uploading model...")
    registered_model = Model.register(model_path=model_folder_path, 
                                      model_name=model_name, 
                                      workspace=ws)

    # create a Conda dependencies environment file
    print("Creating conda dependencies file locally...")
    from azureml.core.conda_dependencies import CondaDependencies 
    mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)
    with open(conda_file,"w") as f:
        f.write(mycondaenv.serialize_to_string())
        
    # create container image configuration
    print("Creating container image configuration...")
    from azureml.core.image import ContainerImage
    image_config = ContainerImage.image_configuration(execution_script = scoring_script_filename,
                                                      runtime = runtime,
                                                      conda_file = conda_file)
    
    # create ACI configuration
    print("Creating ACI configuration...")
    from azureml.core.webservice import AciWebservice, Webservice
    aci_config = AciWebservice.deploy_configuration(
        cpu_cores = cpu_cores, 
        memory_gb = memory_gb, 
        tags = tags, 
        description = description)

    # deploy the webservice to ACI
    print("Deploying webservice to ACI...")
    webservice = Webservice.deploy_from_model(
      workspace=ws, 
      name=service_name, 
      deployment_config=aci_config,
      models = [registered_model], 
      image_config=image_config
    )
    webservice.wait_for_deployment(show_output=True)
    
    return webservice

'''
Your web service which knows how to load the model and use it for scoring needs saved out to a file for the Azure Machine Learning service SDK to deploy it. Run the following cell to create this file.
'''

#%%
get_ipython().run_cell_magic('writefile', '$deployment_folder/scoring-service.py', 'import sys\nimport os\nimport json\nimport numpy as np\nimport pandas as pd\nfrom azureml.core.model import Model\nimport onnxruntime\n\ndef init():\n    global model\n    \n    try:\n        model_path = Model.get_model_path(\'component_compliance\')\n        model_file_path = os.path.join(model_path,\'component_compliance.onnx\')\n        print(\'Loading model from:\', model_file_path)\n        \n        # Load the ONNX model\n        model = onnxruntime.InferenceSession(model_file_path)\n    except Exception as e:\n        print(e)\n        \n# note you can pass in multiple rows for scoring\ndef run(raw_data):\n    try:\n        print("Received input:", raw_data)\n        \n        input_data = np.array(json.loads(raw_data)).astype(np.float32)\n        \n        # Run an ONNX session to classify the input.\n        result = model.run(None, {model.get_inputs()[0].name:input_data})[0] \n        result = result[0][0].item()\n        \n        # return just the classification index (0 or 1)\n        return result\n    except Exception as e:\n        error = str(e)\n        return error')

'''
Next, create your Workspace (or retrieve the existing one if it already exists) and deploy the model as a web service.
Before you run the cell, modify the value of the `service_name` parameter to replace `YOUR-UNIQUE-IDENTIFIER` so that it includes your identifier.
Run the cell to perform the deployment.
'''

# In[34]:
# It is important to change the current working directory so that your generated scoring-service.py is at the root of it. 
# This is required by the Azure Machine Learning SDK
os.chdir(deployment_folder)

# In[35]:
ws =  getOrCreateWorkspace(subscription_id, resource_group, 
                   workspace_name, workspace_region)


# In[36]:
from azureml.core.webservice import Webservice

webservice_name = "complianceservice-zst"

webservice = None
for service in Webservice.list(ws):
    if (service.name == webservice_name):
        webservice = service
        print(webservice.name, webservice)

# Deploy the Model as a WebService

# In[37]:
if webservice == None:
    print(os.getcwd())
    webservice = deployModelAsWebService(ws, model_folder_path=onnx_export_folder, 
                                     model_name="component_compliance", service_name = webservice_name)


# Finally, test your deployed web service.
# In[55]:
# choose a sample from the test data set to send
test_sample = x_test.astype(np.float32)[1502]
test_sample_json = json.dumps(test_sample.tolist())

# invoke the web service
result = webservice.run(input_data=test_sample_json)

result

# You now have a working web service deployed that uses the ONNX version of your Keras deep learning model.
