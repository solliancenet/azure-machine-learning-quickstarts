'''
Azure Machine Learning Pipelines
The goal of this lab is to build a pipeline that demonstrate the basic data science workflow of data preparation, model training, and predictions. Azure Machine Learning, allows you to define distinct steps and make it possible to rerun only the steps you need as you tweak and test your workflow.

In this lab, we will develop a Car Battery Life prediction model using pandas and scikit-learn libraries. The work will be divided into 3 main pipeline steps: (1) preprocess the features, (2) train the model, and (3) make predictions. The preprocess step code will be also reused in the pipeline to make predictions on the test data. Thus you can centralize all your logic for preprocessing data for both train and test.

Using the pipeline steps, we will build and run two pipelines: (1) preprocess training data and train the model, and (2) preprocess the test data and make predictions using the trained model. Each of these pipelines is going to have implicit data dependencies and the example will demonstrate how AML make it possible to reuse previously completed steps and run only the modified or new steps in your pipeline.

The pipelines will be run on the Azure Machine Learning compute.
'''

#Azure Machine Learning and Pipeline SDK-specific Imports
# In[ ]:
import numpy as np
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

'''
Setup
To begin, you will need to provide the following information about your Azure Subscription. 

In the following cell, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments (these values can be acquired from the Azure Portal).

Be sure to replace XXXXX in the values below with your unique identifier.
To get these values, do the following:
1. Navigate to the Azure Portal and login with the credentials provided.
2. From the left hand menu, under Favorites, select `Resource Groups`.
3. In the list, select the resource group with the name similar to `XXXXX`.
4. From the Overview tab, capture the desired values.

Execute the following cell by clicking Run Cell below.
'''

# In[ ]:
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "xxx-xxx-xxx"

#Provide values for the existing Resource Group 
resource_group = "Pipelines-Lablet"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "pipelines-workspace"
workspace_region = "eastus"

# project folder for the script files
project_folder = '.'

'''
Create and connect to an Azure Machine Learning Workspace
Run the following cell to create a new Azure Machine Learning **Workspace** and save the configuration to disk (next to the Jupyter notebook). 

**Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.
'''

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

'''
Create AML Compute Cluster
Azure Machine Learning Compute is a service for provisioning and managing clusters of Azure virtual machines for running machine learning workloads. Let's create a new Aml Compute in the current workspace, if it doesn't already exist. We will run all our pipelines on this compute target.
'''

# In[ ]:
from azureml.core.compute_target import ComputeTargetException

aml_compute_target = "aml-compute"
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


'''
Create the Run Configuration
'''

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
    'scikit-learn',
    'numpy',
    'pandas'
])

'''
The Process Training Data Pipeline Step
The process training data step in the pipeline takes raw training data as input. This data can be a data source that lives in one of the accessible data locations, or intermediate data produced by a previous step in the pipeline. In this example we will upload the raw training data in the workspace's default blob store. Run the following two cells at the end of which we will create a **DataReference** object that points to the raw training data *csv* file stored in the default blob store.
'''

# In[ ]:
# Default datastore (Azure file storage)
def_file_store = ws.get_default_datastore() 
print("Default datastore's name: {}".format(def_file_store.name))
def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))


# In[ ]:
import urllib.request
import os

data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/training-formatted.csv"

# Download the raw training data to your local disk
os.makedirs('./train_data', exist_ok=True)
urllib.request.urlretrieve(data_url, "./train_data/training-raw.csv")

# Upload the raw training data to the blob storage
def_blob_store.upload(src_dir='./train_data', 
                      target_path='battery-life-raw-features', 
                      overwrite=True, 
                      show_progress=True)

raw_train_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="battery_life_raw_data", 
                                      path_on_datastore="battery-life-raw-features/training-raw.csv")
print("DataReference object created")

'''
Create the Process Training Data Pipeline Step
The intermediate data (or output of a Step) is represented by PipelineData object. PipelineData can be produced by one step and consumed in another step by providing the PipelineData object as an output of one step and the input of one or more steps.

Constructing PipelineData
name: *Required* Name of the data item within the pipeline graph
datastore_name: Name of the Datastore to write this output to
output_name: Name of the output
output_mode: Specifies "upload" or "mount" modes for producing output (default: mount)
output_path_on_compute: For "upload" mode, the path to which the module writes this output during execution
output_overwrite: Flag to overwrite pre-existing data

The process training data pipeline step takes the raw_train_data DataReference object as input, and it will output two intermediate PipelineData objects: (1) preprocessor's the step created to process the train data and (2) the processed training data files.

Open preprocess.py in the local machine and examine the arguments, inputs, and outputs for the script. Note that there is an argument called process_mode to distinguish between processing training data vs test data. Reviewing the Python script file will give you a good sense of why the script argument names used below are important.

Review and run the cell below to construct the PipelineData objects and the PythonScriptStep pipeline step:
'''

# In[ ]:
preprocessors = PipelineData('preprocessors', datastore=def_blob_store)
processed_train_data = PipelineData('processed_train_data', datastore=def_blob_store)
print("PipelineData object created")

processTrainDataStep = PythonScriptStep(
    name="process_train_data",
    script_name="preprocess.py", 
    arguments=["--process_mode", 'train',
               "--input", raw_train_data,
               "--preprocessors", preprocessors,
               "--output", processed_train_data],
    inputs=[raw_train_data],
    outputs=[preprocessors, processed_train_data],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("preprocessStep created")

'''
Create the Train Pipeline Step
The train pipeline step takes the *processed_train_data* created in the above step as input and generates another PipelineData object to save the *trained_model* as its output. This is an example of how machine learning pipelines can have many steps and these steps could use or reuse datasources and intermediate data.

Open train.py in the local machine and examine the arguments, inputs, and outputs for the script.
'''

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

'''
Create and Validate the Pipeline
Note that the *trainStep* has implicit data dependency with the *processTrainDataStep* and thus you only include the *trainStep* in your Pipeline object. You will observe that when you run the pipeline that it will first run the **processTrainDataStep** followed by the **trainStep**.
'''

# In[ ]:
pipeline = Pipeline(workspace=ws, steps=[trainStep])
print ("Pipeline is built")

pipeline.validate()
print("Simple validation complete")

'''
Submit the Pipeline
At this point you can run the pipeline and examine the output it produced.
'''

# In[ ]:
pipeline_run = Experiment(ws, 'Sample-ML-Pipeline').submit(pipeline)
print("Pipeline is submitted for execution")

'''
Monitor the Run Details
Run the cell below. 
Log into Azure Portal and observe the order in which the pipeline steps are executed based on their implicit data dependencies.
**processTrainDataStep** followed by the **trainStep**
'''

# In[ ]:
pipeline_run.wait_for_completion(show_output = True)
print()
for child in pipeline_run.get_children():
    print('Pipeline name: {} Pipeline status: {}'.format
    (child.name, child.get_status()))


'''
Make Predictions on Test Data
Create the Raw Test Data DataReference Object
Upload the raw test data in the default blob store and create the **DataReference** object for the raw test data.
'''

# In[ ]:
test_data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/fleet-formatted.csv"

# Download the raw training data to your local disk
os.makedirs('./test_data', exist_ok=True)
urllib.request.urlretrieve(test_data_url, "./test_data/test-raw.csv")

# Upload the raw test data to the blob storage
def_blob_store.upload(src_dir='./test_data', 
                      target_path='battery-life-raw-features', 
                      overwrite=True, 
                      show_progress=True)

raw_test_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="battery_life_raw_test_data", 
                                      path_on_datastore="battery-life-raw-features/test-raw.csv")
print("DataReference object created")

'''
Create the Process Test Data Pipeline Step
Similar to the process train data pipeline step, we create a new step for processing the test data. Note that it is the same script file *preprocess.py* that is used to process both the train and test data. Thus, you can centralize all your logic for preprocessing data for both train and test. The key differences here are: (1) the process_mode is set to *inference*, and (2) the *preprocessors* PipelineData object created during the process train data pipeline step is now an input to this step.

Open preprocess.py in the local machine and examine the arguments, inputs, and outputs for the script. Note that there is an argument called process_mode to distinguish between processing training data vs test data. Reviewing the Python script file will give you a good sense of why the script argument names used below are important.
'''

# In[ ]:
processed_test_data = PipelineData('processed_test_data', datastore=def_blob_store)
print("PipelineData object created")

processTestDataStep = PythonScriptStep(
    name="process_test_data",
    script_name="preprocess.py", 
    arguments=["--process_mode", 'inference',
               "--input", raw_test_data,
               "--preprocessors", preprocessors,
               "--output", processed_test_data],
    inputs=[raw_test_data, preprocessors],
    outputs=[processed_test_data],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=project_folder
)
print("preprocessStep created")

'''
Create the Inference Pipeline Step to Make Predictions
The inference pipeline step takes the *processed_test_data* created in the above step and the *trained_model* created in the train step as two inputs and generates *inference_output* as its output. This is yet another example of how machine learning pipelines can have many steps and these steps could use or reuse datasources and intermediate data.
Open inference.py in the local machine and examine the arguments, inputs, and outputs for the script.
'''

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

'''
Create and Validate the Pipeline
Note that the *inferenceStep* has implicit data dependency with **ALL** of the previous pipeline steps.
'''

# In[ ]:
inference_pipeline = Pipeline(workspace=ws, steps=[inferenceStep])
print ("Inference Pipeline is built")

inference_pipeline.validate()
print("Simple validation complete")

'''
Submit the Pipeline
At this point you can run the pipeline and examine the output it produced. This pipeline will reuse the previous run's output for both the **process_train_data** and **train** pipeline steps.

This example demonstrates how AML make it possible to reuse previously completed steps and run only the modified or new steps in your pipeline.
'''

# In[ ]:
inference_pipeline_run = Experiment(ws, 'Sample-ML-Pipeline').submit(inference_pipeline)
print("Inference pipeline is submitted for execution")

'''
Monitor the Run Details
Run the cell below.
Log into Azure Portal and observe the order in which the pipeline steps are executed based on their implicit data dependencies.
'''

# In[ ]:
# Wait for all child runs are completed
inference_pipeline_run.wait_for_completion(show_output = True)
print()
for child in inference_pipeline_run.get_children():
    print('Pipeline name: {} Pipeline status: {}'.format
    (child.name, child.get_status()))

'''
Download and Observe the Predictions
'''

# In[ ]:
# access the inference_output
data = inference_pipeline_run.find_step_run('inference')[0].get_output_data('inference_output')
# download the predictions to local path
data.download('.', show_progress=True)
# print the predictions
predictions = np.loadtxt(os.path.join('./', data.path_on_datastore, 'results.txt'))
print(predictions)
