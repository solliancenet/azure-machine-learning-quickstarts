# Automated Machine Learning with Azure Machine Learning
# 
# **Please complete Part 1 before continuing**
# 
# In this part, you will test, register, and deploy the model you trained in the first part using the Azure ML Python SDK.

# In[ ]:
import os
import numpy as np
import pandas as pd
import logging

from azureml.core import Workspace, Experiment, Run
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core.model import Model

# Setup
# To begin, you will need to provide the following information about your Azure Subscription. 
# 
# In the following cell, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name`, `workspace_region`, and `experiment_name` as directed by the comments (*these values can be acquired from the Azure Portal*). Note that values need to be consistent with those used in part 1.
# 
# **Be sure to replace XXXXX in the values below with your unique identifier.**
# 
# To get these values, do the following:
# 1. Navigate to the Azure Portal and login with the credentials provided.
# 2. From the left hand menu, under Favorites, select `Resource Groups`.
# 3. In the list, select the resource group with the name similar to `XXXXX`.
# 4. From the Overview tab, capture the desired values.

# In[ ]:
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "xxx-xxx-xxx"

#Provide values for the existing Resource Group 
resource_group = "Quick-Starts-Labs"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "auto-ml-demo"
workspace_region = "eastus"

experiment_name = "auto-ml-exp"

scripts_folder = './automl_scripts'
automl_scoring_service_name = 'automl_scoring_service.py'

# Connect to the Azure Machine Learning Workspace
# Run the following cell to connect the Azure Machine Learning **Workspace** created in part 1.
# **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.

# In[ ]:
ws = Workspace.get(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group)

# Get the Fitted Model from the AutoML Run Object
# Run the following cell to access the best model from the AutoML experiment completed in part 1

# In[ ]:
experiment = Experiment(ws, experiment_name)
automl_run = AutoMLRun(experiment, list(experiment.get_runs())[0].id)
best_run, fitted_model = automl_run.get_output()
print(fitted_model)

# Test Model
# Generate test data and run prediction on the model.

# In[ ]:
columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', 
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']
test_data = [[1, 3, 10, 15, 4, 27, 7, 'None', False, 0, 2.0, 1.0, 80], 
        [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]
test_data_df = pd.DataFrame(data, columns = columns)

# In[ ]:
fitted_model.predict(test_data_df)

# Register Model

# In[ ]:
registered_model = automl_run.register_model(description="This model was trained using automl.", 
                                             tags={"type": "regression", "run_id": automl_run.id})
print('Registered model name: ', registered_model.name)

# Deploy Model

# Create the Scoring Script
# Run the next two cells to create the scoring script on local disk

# In[ ]:
# Create the scripts folder locally
if not os.path.exists(scripts_folder):
    os.makedirs(scripts_folder)

# **Important** Please update the `model_name` variable in the script below. The model name should be the same as the `Registered model name` printed above.

# In[ ]:
get_ipython().run_cell_magic('writefile', '$scripts_folder/$automl_scoring_service_name', "\nimport json\nimport numpy as np\nimport pandas as pd\nimport azureml.train.automl as AutoML\n\ncolumns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', \n           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', \n           'precipDepth', 'temperature']\n\ndef init():\n    try:\n        # One-time initialization of predictive model and scaler\n        from azureml.core.model import Model\n        from sklearn.externals import joblib\n        global model\n        \n        model_name = 'xxx'\n        print('Looking for model path for model: ', model_name)\n        model_path = Model.get_model_path(model_name=model_name)\n        print('Looking for model in: ', model_path)\n        model = joblib.load(model_path)\n        print('Model loaded...')\n\n    except Exception as e:\n        print('Exception during init: ', str(e))\n\ndef run(input_json):     \n    try:\n        inputs = json.loads(input_json)\n        data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)\n        # Get the predictions...\n        prediction = model.predict(data_df).tolist()\n        prediction = json.dumps(prediction)\n    except Exception as e:\n        prediction = str(e)\n    return prediction")

# Package Model

# In[ ]:
# create a Conda dependencies environment file
print("Creating conda dependencies file locally...")
from azureml.core.conda_dependencies import CondaDependencies 
conda_packages = ['numpy', 'pandas', 'scikit-learn']
pip_packages = ['azureml-sdk[automl]']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

os.chdir(scripts_folder)

conda_file = 'automl_dependencies.yml'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

runtime = 'python'

# create container image configuration
print("Creating container image configuration...")
from azureml.core.image import ContainerImage
image_config = ContainerImage.image_configuration(execution_script = automl_scoring_service_name, 
                                                  runtime = runtime, conda_file = conda_file)

# create the image
image_name = 'nyc-taxi-automl-image'

from azureml.core import Image
image = Image.create(name=image_name, models=[registered_model], image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)

os.chdir("..")

# Deploy Azure Container Instance (ACI) Webservice

# In[ ]:
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

# Test Deployment
# Make direct calls on the service object

# In[ ]:
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

# Make HTTP calls to test the Webservice

# In[ ]:
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
