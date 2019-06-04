
# # Azure Machine Learning Pipelines
# 
# The goal of this quickstart is to build a pipeline that demonstrate the basic data science workflow of data preparation, model training, and predictions. Azure Machine Learning, allows you to define distinct steps and make it possible to rerun only the steps you need as you tweak and test your workflow.
# 
# In this quickstart, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/). The data is enriched with holiday and weather data. The goal is to train a regression model to predict taxi fares in New York City based on input features such as, number of passengers, trip distance, datetime, holiday information and weather information.
# 
# The machine learning pipeline in this quickstart is organized into three steps:
# 
# - **Preprocess Training and Input Data:** We want to preprocess the data to better represent the datetime features, such as hours of the day, and day of the week to capture the cyclical nature of these features.
# 
# - **Model Training:** We will use data transformations and the GradientBoostingRegressor algorithm from the scikit-learn library to train a regression model. The trained model will be saved for later making predictions.
# 
# - **Model Inference:** In this scenario, we want to support **bulk predictions**. Each time an input file is uploaded to a data store, we want to initiate bulk model predictions on the input data. However, before we do model predictions, we will reuse the preprocess data step to process input data, and then make predictions on the processed data using the trained model.
# 
# Each of these pipelines is going to have implicit data dependencies and the example will demonstrate how AML make it possible to reuse previously completed steps and run only the modified or new steps in your pipeline.
# 
# **The pipelines will be run on the Azure Machine Learning compute.**

# ### Azure Machine Learning and Pipeline SDK-specific Imports

# In[ ]:


import numpy as np
import pandas as pd
import azureml.core
from azureml.core import Workspace, Experiment, Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.widgets import RunDetails

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
print("Pipeline SDK-specific imports completed")


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

aml_compute_target = "aml-compute"

experiment_name = 'quick-starts-pipeline'

# project folder for the script files
project_folder = 'aml-pipelines-scripts'
data_location = 'aml-pipelines-data'
test_data_location = 'aml-pipelines-test-data'


# In[ ]:


#Provide the Subscription ID of your existing Azure subscription
subscription_id = "" # <- needs to be the subscription with the Quick-Starts resource group

subscription_id = 'fdbba0bc-f686-4b8b-8b29-394e0d9ae697'

#Provide values for the existing Resource Group 
resource_group = "Quick-Starts-XXXXX" # <- replace XXXXX with your unique identifier

resource_group = "Quick-Starts-Labs"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "quick-starts-ws-XXXXX" # <- replace XXXXX with your unique identifier
workspace_region = "eastus" # <- region of your Quick-Starts resource group

workspace_name = "quick-starts"


# ### Download the training data and code for the pipeline steps
# 
# Once you download the files, please review the code for the three pipeline steps in the following files: `preprocess.py`, `train.py`, and `inference.py`

# In[ ]:


import urllib.request
import os

data_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
            'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
            'quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv')

preprocess_script = ('https://quickstartsws9073123377.blob.core.windows.net/'
                     'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
                     'quickstarts/pipeline-scripts/preprocess.py')
                     
train_script = ('https://quickstartsws9073123377.blob.core.windows.net/'
                'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
                'quickstarts/pipeline-scripts/train.py')
                
inference_script = ('https://quickstartsws9073123377.blob.core.windows.net/'
                    'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
                    'quickstarts/pipeline-scripts/inference.py')


# Download the raw training data to your local disk
os.makedirs(data_location, exist_ok=True)
urllib.request.urlretrieve(data_url, os.path.join(data_location, 'nyc-taxi-sample-data.csv'))

# Download the script files to your local disk
os.makedirs(project_folder, exist_ok=True)
urllib.request.urlretrieve(preprocess_script, os.path.join(project_folder, 'preprocess.py'))
urllib.request.urlretrieve(train_script, os.path.join(project_folder, 'train.py'))
urllib.request.urlretrieve(inference_script, os.path.join(project_folder, 'inference.py'))


# ### Create and connect to an Azure Machine Learning Workspace
# 
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
# 
# Azure Machine Learning Compute is a service for provisioning and managing clusters of Azure virtual machines for running machine learning workloads. Let's create a new Aml Compute in the current workspace, if it doesn't already exist. We will run all our pipelines on this compute target.

# In[ ]:


from azureml.core.compute_target import ComputeTargetException

try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 1)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
print("Aml Compute attached")


# ### Create the Run Configuration

# In[ ]:


from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above. 
run_amlcompute.target = aml_compute_target

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False

# Auto-prepare the Docker image when used for execution (if it is not already prepared)
run_amlcompute.auto_prepare_environment = True

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'numpy',
    'pandas',
    'scikit-learn',
    'sklearn_pandas'
])


# ### The Process Training Data Pipeline Step

# The process training data step in the pipeline takes raw training data as input. This data can be a data source that lives in one of the accessible data locations, or intermediate data produced by a previous step in the pipeline. In this example we will upload the raw training data in the workspace's default blob store. Run the following two cells at the end of which we will create a **DataReference** object that points to the raw training data *csv* file stored in the default blob store.

# In[ ]:


# Default datastore (Azure file storage)
def_file_store = ws.get_default_datastore() 
print("Default datastore's name: {}".format(def_file_store.name))
def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))


# In[ ]:


# Upload the raw training data to the blob storage
def_blob_store.upload(src_dir=data_location, 
                      target_path='nyc-taxi-raw-features', 
                      overwrite=True, 
                      show_progress=True)

raw_train_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="nyc_taxi_raw_features", 
                                      path_on_datastore="nyc-taxi-raw-features/nyc-taxi-sample-data.csv")
print("DataReference object created")


# ### Create the Process Training Data Pipeline Step

# The intermediate data (or output of a Step) is represented by PipelineData object. PipelineData can be produced by one step and consumed in another step by providing the PipelineData object as an output of one step and the input of one or more steps.
# 
# The process training data pipeline step takes the raw_train_data DataReference object as input, and it will output an intermediate PipelineData object that holds the processed training data with the new engineered features for datetime components: hour of the day, and day of the week.
# 
# Review and run the cell below to construct the PipelineData objects and the PythonScriptStep pipeline step:
# 
# *Open preprocess.py in the local machine and examine the arguments, inputs, and outputs for the script. Note that there is an argument called process_mode to distinguish between processing training data vs test data. Reviewing the Python script file will give you a good sense of why the script argument names used below are important.*

# In[ ]:


processed_train_data = PipelineData('processed_train_data', datastore=def_blob_store)
print("PipelineData object created")

processTrainDataStep = PythonScriptStep(
    name="process_train_data",
    script_name="preprocess.py", 
    arguments=["--process_mode", 'train',
               "--input", raw_train_data,
               "--output", processed_train_data],
    inputs=[raw_train_data],
    outputs=[processed_train_data],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("preprocessStep created")


# ### Create the Train Pipeline Step

# The train pipeline step takes the *processed_train_data* created in the above step as input and generates another PipelineData object to save the *trained_model* as its output. This is an example of how machine learning pipelines can have many steps and these steps could use or reuse datasources and intermediate data.
# 
# *Open train.py in the local machine and examine the arguments, inputs, and outputs for the script.*

# In[ ]:


trained_model = PipelineData('trained_model', datastore=def_blob_store)
print("PipelineData object created")

trainStep = PythonScriptStep(
    name="train",
    script_name="train.py", 
    arguments=["--input", processed_train_data, "--output", trained_model],
    inputs=[processed_train_data],
    outputs=[trained_model],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("trainStep created")


# ### Create and Validate the Pipeline
# 
# Note that the *trainStep* has implicit data dependency with the *processTrainDataStep* and thus you only include the *trainStep* in your Pipeline object. You will observe that when you run the pipeline that it will first run the **processTrainDataStep** followed by the **trainStep**.

# In[ ]:


pipeline = Pipeline(workspace=ws, steps=[trainStep])
print ("Pipeline is built")

pipeline.validate()
print("Simple validation complete")


# ### Submit the Pipeline
# 
# At this point you can run the pipeline and examine the output it produced.

# In[ ]:


pipeline_run = Experiment(ws, experiment_name).submit(pipeline)
print("Pipeline is submitted for execution")


# ### Monitor the Run Details
# 
# Observe the order in which the pipeline steps are executed: **processTrainDataStep** followed by the **trainStep**
# 
# Wait till both steps finish running. The cell below should periodically auto-refresh and you can also rerun the cell to force a refresh.

# In[ ]:


RunDetails(pipeline_run).show()


# ## Bulk Predictions

# ### Create the Raw Test Data DataReference Object
# 
# Create the **DataReference** object where the raw bulk test or input data file will be uploaded.

# In[ ]:


# Upload dummy raw test data to the blob storage
os.makedirs(test_data_location, exist_ok=True)
pd.DataFrame([[0]], columns = ['col1']).to_csv(os.path.join(test_data_location, 'raw-test-data.csv'), header=True, index=False)

def_blob_store.upload(src_dir=test_data_location, 
                      target_path='bulk-test-data', 
                      overwrite=True, 
                      show_progress=True)

# Create a DataReference object referencing the 'raw-test-data.csv' file
raw_bulk_test_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="raw_bulk_test_data", 
                                      path_on_datastore="bulk-test-data/raw-test-data.csv")
print("DataReference object created")


# ### Create the Process Test Data Pipeline Step

# Similar to the process train data pipeline step, we create a new step for processing the test data. Note that it is the same script file *preprocess.py* that is used to process both the train and test data. Thus, you can centralize all your logic for preprocessing data for both train and test. The key differences here is that the process_mode argument is set to *inference*. 
# 
# *Open preprocess.py in the local machine and examine the arguments, inputs, and outputs for the script. Note that there is an argument called process_mode to distinguish between processing training data vs test data. Reviewing the Python script file will give you a good sense of why the script argument names used below are important.*

# In[ ]:


processed_test_data = PipelineData('processed_test_data', datastore=def_blob_store)
print("PipelineData object created")

processTestDataStep = PythonScriptStep(
    name="process_test_data",
    script_name="preprocess.py", 
    arguments=["--process_mode", 'inference',
               "--input", raw_bulk_test_data,
               "--output", processed_test_data],
    inputs=[raw_bulk_test_data],
    outputs=[processed_test_data],
    allow_reuse = False,
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("preprocessStep created")


# ### Create the Inference Pipeline Step to Make Predictions

# The inference pipeline step takes the *processed_test_data* created in the above step and the *trained_model* created in the train step as two inputs and generates *inference_output* as its output. This is yet another example of how machine learning pipelines can have many steps and these steps could use or reuse datasources and intermediate data.
# 
# *Open inference.py in the local machine and examine the arguments, inputs, and outputs for the script.*

# In[ ]:


inference_output = PipelineData('inference_output', datastore=def_blob_store)
print("PipelineData object created")

inferenceStep = PythonScriptStep(
    name="inference",
    script_name="inference.py", 
    arguments=["--input", processed_test_data,
               "--model", trained_model,
               "--output", inference_output],
    inputs=[processed_test_data, trained_model],
    outputs=[inference_output],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("inferenceStep created")


# ### Create and Validate the Pipeline
# 
# Note that the *inferenceStep* has implicit data dependency with **ALL** of the previous pipeline steps.

# In[ ]:


inference_pipeline = Pipeline(workspace=ws, steps=[inferenceStep])
print ("Inference Pipeline is built")

inference_pipeline.validate()
print("Simple validation complete")


# ### Publish the Inference Pipeline
# 
# Note that we are not submitting the pipeline to run, instead we are publishing the pipeline. When you publish a pipeline, it can be submitted to run without the Python code which constructed the Pipeline.

# In[ ]:


pipeline_name = 'Inference Pipeline'
published_pipeline = inference_pipeline.publish(name = pipeline_name)


# ### Schedule the Inference Pipeline
# 
# We want to run the Inference Pipeline when a new test or input data is uploaded at the location referenced by the `raw_bulk_test_data` DataReference object. The next cell creates a Schedule to monitor the datastore for changes, and is responsible for running the `Inference Pipeline` when it detects a new file being uploaded.
# 
# **Please disable the Schedule after completing the quickstart. To disable the Schedule, run the `Cleanup Resources` section at the end of the notebook.**

# In[ ]:


from azureml.pipeline.core.schedule import Schedule

schedule = Schedule.create(workspace=ws, name=pipeline_name + "_sch",
                           pipeline_id=published_pipeline.id, 
                           experiment_name=experiment_name,
                           datastore=def_blob_store,
                           wait_for_provisioning=True,
                           description="Datastore scheduler for Pipeline: " + pipeline_name,
                           path_on_datastore='bulk-test-data',
                           polling_interval=1 # in minutes
                           )

print("Created schedule with id: {}".format(schedule.id))


# In[ ]:


# Check the status of the Schedule and confirm it's Active
print('Schedule status: ', schedule.status)


# ### Test the Inference Pipeline
# 
# Create some test data to make bulk predictions. Upload the test data to the `bulk-test-data` blob store. 

# In[ ]:


# Download the raw training data to your local disk
columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', 
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']

data = [[1, 4, 10, 15, 4, 5, 7, 'None', False, 0, 0.0, 0.0, 80], 
        [1, 1, 5, 6, 0, 20, 1, 'Martin Luther King, Jr. Day', True, 0, 2.0, 3.0, 35]]

data_df = pd.DataFrame(data, columns = columns)

os.makedirs(test_data_location, exist_ok=True)
data_df.to_csv(os.path.join(test_data_location, 'raw-test-data.csv'), header=True, index=False)

from datetime import datetime
data_upload_time = datetime.utcnow()
print('Data upload time in UTC: ', data_upload_time)

# Upload the raw test data to the blob storage
def_blob_store.upload(src_dir=test_data_location, 
                      target_path='bulk-test-data', 
                      overwrite=True, 
                      show_progress=True)

# Wait for 65 seconds...
import time
print('Please wait...')
time.sleep(65)
print('Done!')


# ### Wait for Schedule to Trigger
# 
# Schedule's polling interval is 1 minute. You can also log into Azure Portal and navigate to your `resource group -> workspace -> experiment` to see if the `Inference Pipeline` has started executing.
# 
# **If the inference_pipeline_run object in the below cell is None, it means that the Schedule has not triggered yet!**
# 
# **If the Schedule does not trigger in 2 minutes, try rerunning the data upload cell again!**

# In[ ]:


# Confirm that the inference_pipeline_run object is NOT None
inference_pipeline_run = schedule.get_last_pipeline_run()
print(inference_pipeline_run)


# *If upload the test data file more than once, confirm that we have the latest pipeline run object. We will compare the pipeline start time with the time you uploaded the test data file.*

# In[ ]:


# confirm the start time
import dateutil.parser

if inference_pipeline_run.get_details()['status'] != 'NotStarted':
    pipeline_starttime = dateutil.parser.parse(inference_pipeline_run.get_details()['startTimeUtc'], ignoretz=True)
else:
    pipeline_starttime = datetime.utcnow()

if(pipeline_starttime > data_upload_time):
    print('We have the correct inference pipeline run! Proceed to next cell.')
else:
    print('Rerun the above cell to get the latest inference pipeline run!')


# ### Monitor the Run Details
# 
# Observe the order in which the pipeline steps are executed based on their implicit data dependencies.
# 
# Wait till all steps finish running. The cell below should periodically auto-refresh and you can also rerun the cell to force a refresh.
# 
# **This example demonstrates how AML make it possible to reuse previously completed steps and run only the modified or new steps in your pipeline.**

# In[ ]:


RunDetails(inference_pipeline_run).show()


# ### Download and Observe the Predictions

# In[ ]:


# access the inference_output
data = inference_pipeline_run.find_step_run('inference')[0].get_output_data('inference_output')
# download the predictions to local path
data.download('.', show_progress=True)
# print the predictions
predictions = np.loadtxt(os.path.join('./', data.path_on_datastore, 'results.txt'))
print(predictions)


# ### Cleanup Resources
# 
# If you are done experimenting with this quickstart, run the following two cell to clean up resources.

# In[ ]:


schedule.disable()


# In[ ]:


aml_compute.delete()


# In[ ]:




