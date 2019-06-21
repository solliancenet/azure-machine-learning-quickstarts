# # Automated Machine Learning with Azure Machine Learning Service

# ## Quick Start Overview
# 
# In this quickstart we will be building a regression model to predict **Taxi Fares in New York City**. We will download a preprocessed labeled training data with features such as number of passengers, trip distance, datetime, holiday information and weather information.
# 
# You will use compute resources provided by Azure Machine Learning (AML) to **remotely** train a **set** of models using **Automated Machine Learning**, evaluate performance of each model and pick the best performing model to deploy as a web service hosted by **Azure Container Instance**.
# 
# Because you will be using the Azure Machine Learning SDK, you will be able to provision all your required Azure resources directly from this notebook, without having to use the Azure Portal to create any resources.

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

# In[10]:


#Provide the Subscription ID of your existing Azure subscription
subscription_id = "" # <- needs to be the subscription with the Quick-Starts resource group

#Provide values for the existing Resource Group 
resource_group = "Quick-Starts-XXXXX" # <- replace XXXXX with your unique identifier

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "quick-starts-ws-XXXXX" # <- replace XXXXX with your unique identifier
workspace_region = "eastus" # <- region of your Quick-Starts resource group


# Constants, you can leave these values as they are or experiment with changing them after you have completed the notebook once

# In[11]:


experiment_name = 'automl-regression'
project_folder = './automl-regression'

# this is the URL to the CSV file containing the training and test data
data_url = ('https://mlopswsstorage9733ce347d.blob.core.windows.net/'
            'azureml-blobstore-4a000316-c93b-407f-b246-ed8471c34db8/'
            'nyc-taxi-sample-data/nyc-taxi-sample-data.csv')

cluster_name = "cpucluster"


# ### Import required packages

# The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:
# - Creating a **Workspace** that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
# - Creating **Experiments** in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
# - Creating **Compute** resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU. 
# - Using **Automated Machine Learning (AutoML)** to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify. 
# - Packaging a Docker **Image** that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
# - Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the **Web Service**.
# 
# In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following cell to do so:

# In[12]:


import logging
import os
import random
import re

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import Image
from azureml.core.model import Model
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core import Workspace


# ### Create and connect to an Azure Machine Learning Workspace
# 
# Run the following cell to create a new Azure Machine Learning **Workspace**.
# 
# **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.

# In[13]:


ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

ws.write_config()

print('Workspace configuration succeeded')


# ## Create a Workspace Experiment

# Notice in the first line of the cell below, we can re-load the config we saved previously and then display a summary of the environment.

# In[34]:


ws = Workspace.from_config()

# Display a summary of the current environment 
output = {}
output['SDK version'] = azureml.core.VERSION
#output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
pd.set_option('display.max_colwidth', -1)
pd.DataFrame(data=output, index=['']).T


# Next, create a new Experiment. 

# In[15]:


experiment = Experiment(ws, experiment_name)


# ## Remotely train multiple models using Auto ML and Azure ML Compute

# In the following cells, you will *not* train the model against the data you just downloaded using the resources provided by Azure Notebooks. Instead, you will deploy an Azure ML Compute cluster that will download the data and use Auto ML to train multiple models, evaluate the performance and allow you to retrieve the best model that was trained. In other words, all of the training will be performed remotely with respect to this notebook. 
# 
# 
# As you will see this is almost entirely done thru configuration, with very little code required. 

# ### Create the data loading script for remote compute

# The Azure Machine Learning Compute cluster needs to know how to get the data to train against. You can package this logic in a script that will be executed by the compute when it starts executing the training.
# 
# Run the following cells to locally create the **get_data.py** script that will be deployed to remote compute. You will also use this script when you want train the model locally. 
# 
# Observe that the get_data method returns the features (`X`) and the labels (`y`) in an object. This structure is expected later when you will configure Auto ML.

# In[16]:


# create project folder
if not os.path.exists(project_folder):
    os.makedirs(project_folder)


# In[35]:


get_ipython().run_cell_magic('writefile', '$project_folder/get_data.py', '\nimport pandas as pd\n\ndef get_data():\n    \n    data_url = (\'https://mlopswsstorage9733ce347d.blob.core.windows.net/\'\n            \'azureml-blobstore-4a000316-c93b-407f-b246-ed8471c34db8/\'\n            \'nyc-taxi-sample-data/nyc-taxi-sample-data.csv\')\n    \n    df = pd.read_csv(data_url)\n    x_df = df.drop([\'totalAmount\'], axis=1).values\n    y_df = df[\'totalAmount\'].values.flatten()\n    \n    return {"X": x_df, "y": y_df}')


# ### Create AML Compute Cluster
# 
# Now you are ready to create the compute cluster. Run the following cell to create a new compute cluster (or retrieve the existing cluster if it already exists). The code below will create a *CPU based* cluster where each node in the cluster is of the size `STANDARD_D12_V2`, and the cluster will have at most *4* such nodes. 

# In[18]:


### Create AML CPU based Compute Cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D12_V2',
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use the 'status' property to get a detailed status for the current AmlCompute. 
print(compute_target.status.serialize())


# ### Instantiate an Automated ML Config
# 
# Run the following cell to configure the Auto ML run. In short what you are configuring here is the training of a regressor model that will attempt to predict the value of the target (`totalAmount`) based on all the other features in the data set. The run is configured to try at most 5 iterations where no iteration can run longer that 2 minutes. 
# 
# Additionally, the data will be automatically pre-processed in different ways as a part of the automated model training (as indicated by the `preprocess` attribute having a value of `True`). This is a very powerful feature of Auto ML as it tries many best practices approaches for you, and saves you a lot of time and effort in the process.
# 
# The goal of Auto ML in this case is to find the best models that result, as measure by the normalized root mean squared error metric (as indicated by the `primary_metric` attribute). The error is basically a measure of what the model predicts versus what was provided as the "answer" in the training data. In short, AutoML will try to get the error as low as possible when trying its combination of approaches.  
# 
# The local path to the script you created to retrieve the data is supplied to the AutoMLConfig, ensuring the file is made available to the remote cluster. The actual execution of this training will occur on the compute cluster you created previously. 
# 
# In general, the AutoMLConfig is very flexible, allowing you to specify all of the following:
# - Task type (classification, regression, forecasting)
# - Number of algorithm iterations and maximum time per iteration
# - Accuracy metric to optimize
# - Algorithms to blacklist (skip)/whitelist (include)
# - Number of cross-validations
# - Compute targets
# - Training data
# 
# Run the following cell to create the configuration.

# In[36]:


automl_config = AutoMLConfig(task="regression",
                             iterations=5,
                             iteration_timeout_minutes = 2, 
                             max_cores_per_iteration = 10,
                             primary_metric="normalized_root_mean_squared_error",
                             preprocess=True,
                             n_cross_validations = 3,
                             debug_log = 'automl.log',
                             verbosity = logging.DEBUG,
                             data_script = project_folder + "/get_data.py",
                             path = project_folder
                            )


# ## Run locally
# 
# You can run AutomML locally, that is it will use the resource provided by your Azure Notebook environment. 
# 
# Run the following cell to run the experiment locally. Note this will take **a few minutes**.

# In[37]:


local_run = experiment.submit(automl_config, show_output=True)


# ### Run our Experiment on AML Compute
# 
# Let's increase the performance by performing the training the AML Compute cluster. This will remotely train multiple models, evaluate them and allow you review the performance characteristics of each one, as well as to pick the *best model* that was trained and download it. 
# 
# We will alter the configuration slightly to perform more iterations. Run the following cell to execute the experiment on the remote compute cluster.

# In[22]:


automl_config = AutoMLConfig(task = 'regression',
                             iterations = 5,
                             iteration_timeout_minutes = 2, 
                             max_cores_per_iteration = 10,
                             preprocess= True,
                             primary_metric='normalized_root_mean_squared_error',
                             n_cross_validations = 3,
                             debug_log = 'automl.log',
                             verbosity = logging.DEBUG,
                             data_script = project_folder + "/get_data.py",
                             compute_target = compute_target,
                             path = project_folder)
remote_run = experiment.submit(automl_config, show_output=False)
remote_run


# Once the above cell completes, the run is starting but will likely have a status of `Preparing` for you. To wait for the run to complete before continuing (and to view the training status updates as they happen), run the following cell. The first time you run this, it will take **about 12-15 minutes** to complete as the cluster is configured and then the AutoML job is run. Output will be streamed as the tasks progress):

# In[23]:


remote_run.wait_for_completion(show_output=True)


# ### Get the best run and the trained model
# 
# At this point you have multiple runs, each with a different trained models. How can you get the model that performed the best? Run the following cells to learn how.

# In[24]:


best_run, fitted_model = remote_run.get_output()
print(best_run)
print(fitted_model)


# At this point you now have a model you could use for predicting NYC Taxi Fares. You would typically use this model in one of two ways:
# - Use the model file within other notebooks to batch score predictions.
# - Deploy the model file as a web service that applications can call. 
# 
# In the following, you will explore the latter option to deploy the best model as a web service.

# ## Download the best model 
# With a run object in hand, it is trivial to download the model. 

# In[25]:


# fetch the best model
best_run.download_file("outputs/model.pkl",
                       output_file_path = "./model.pkl")


# ## Register Model
# 
# Azure Machine Learning provides a Model Registry that acts like a version controlled repository for each of your trained models. To version a model, you use  the SDK as follows. Run the following cell to register the best model with Azure Machine Learning. 

# In[26]:


# register the model for deployment
model = Model.register(model_path = "./model.pkl", # this points to a local file
                       model_name = "nyc-taxi-automl-predictor", # name the model is registered as
                       tags = {'area': "auto", 'type': "regression"}, 
                       description = "NYC Taxi Fare Predictor", 
                       workspace = ws)

print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))


# ## Deploy the Model as a Web Service

# ### Create the Scoring Script
# 
# Azure Machine Learning SDK gives you control over the logic of the web service, so that you can define how it retrieves the model and how the model is used for scoring. This is an important bit of flexibility. For example, you often have to prepare any input data before sending it to your model for scoring. You can define this data preparation logic (as well as the model loading approach) in the scoring file. 
# 
# Run the following cell to create a scoring file that will be included in the Docker Image that contains your deployed web service.
# 
# **Important** Please update the `model_name` variable in the script below. The model name should be the same as the `Registered model name` printed above.

# In[29]:


get_ipython().run_cell_magic('writefile', 'scoring_service.py', "\nimport json\nimport numpy as np\nimport azureml.train.automl as AutoML\n\ncolumns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', \n           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', \n           'precipDepth', 'temperature']\n\ndef init():\n    try:\n        # One-time initialization of predictive model and scaler\n        from azureml.core.model import Model\n        from sklearn.externals import joblib\n        global model\n        \n        model_name = 'nyc-taxi-automl-predictor'\n        print('Looking for model path for model: ', model_name)\n        model_path = Model.get_model_path(model_name=model_name)\n        print('Looking for model in: ', model_path)\n        model = joblib.load(model_path)\n        print('Model loaded...')\n\n    except Exception as e:\n        print('Exception during init: ', str(e))\n\ndef run(input_json):     \n    try:\n        inputs = json.loads(input_json)\n        # Get the predictions...\n        prediction = model.predict(np.array(inputs).reshape(-1, len(columns))).tolist()\n        prediction = json.dumps(prediction)\n    except Exception as e:\n        prediction = str(e)\n    return prediction")


# ## Package Model

# In[30]:


# create a Conda dependencies environment file
print("Creating conda dependencies file locally...")
from azureml.core.conda_dependencies import CondaDependencies 
conda_packages = ['numpy', 'scikit-learn']
pip_packages = ['azureml-sdk[automl]']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

conda_file = 'automl_dependencies.yml'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

runtime = 'python'

# create container image configuration
print("Creating container image configuration...")
from azureml.core.image import ContainerImage
image_config = ContainerImage.image_configuration(execution_script = 'scoring_service.py', 
                                                  runtime = runtime, conda_file = conda_file)

# create the image
image_name = 'nyc-taxi-automl-image'

from azureml.core import Image
image = Image.create(name=image_name, models=[model], image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)


# ## Deploy Model to Azure Container Instance (ACI) as a Web Service

# In[31]:


from azureml.core.webservice import AciWebservice, Webservice

aci_name = 'automl-aci-cluster01'

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name': aci_name}, 
    description = 'NYC Taxi Fare Predictor Web Service')

service_name = 'nyc-taxi-automl-service'

aci_service = Webservice.deploy_from_image(deployment_config=aci_config, 
                                           image=image, 
                                           name=service_name, 
                                           workspace=ws)

aci_service.wait_for_deployment(show_output=True)


# ## Test the deployed web service

# ### Make direct calls on the service object

# In[32]:


import json

data1 = [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]

data2 = [[1, 3, 10, 15, 4, 27, 7, 'None', False, 0, 2.0, 1.0, 80], 
         [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

result = aci_service.run(json.dumps(data1))
print('Predictions for data1')
print(result)

result = aci_service.run(json.dumps(data2))
print('Predictions for data2')
print(result)


# ### Make HTTP calls to test the deployed Web Service

# In[33]:


import requests

url = aci_service.scoring_uri
print('ACI Service: {} scoring URI is: {}'.format(service_name, url))
headers = {'Content-Type':'application/json'}

response = requests.post(url, json.dumps(data1), headers=headers)
print('Predictions for data1')
print(response.text)
response = requests.post(url, json.dumps(data2), headers=headers)
print('Predictions for data2')
print(response.text)


# In[ ]:




