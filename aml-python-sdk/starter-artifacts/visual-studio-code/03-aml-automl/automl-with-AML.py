# # Automated Machine Learning with Azure Machine Learning
# 
# **Please complete Part 1 before continuing**
# 
# In this part, you will test, register, and deploy the model you trained in the first part using the Azure ML Python SDK.

# In[ ]:

import os
import numpy as np
import pandas as pd
import logging

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice


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


# ## Connect to the Azure Machine Learning Workspace
# 
# Run the following cell to connect the Azure Machine Learning **Workspace** created in part 1.
# 
# **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.

# In[ ]:

ws = Workspace.get(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group)


# ## Access the Deployed Webservice from the Workspace

# In[ ]:

service_name = 'nyc-taxi-predictor'
webservice = ws.webservices[service_name]


# ## Test Deployment

# ### Make direct calls on the service object

# In[ ]:

import json

data1 = [[1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

data2 = [[1, 3, 10, 15, 4, 27, 7, 'None', False, 0, 2.0, 1.0, 80], 
         [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

result = webservice.run(json.dumps({'data': data1}))
print('Predictions for data1')
print(result)

result = webservice.run(json.dumps({'data': data2}))
print('Predictions for data2')
print(result)


# ### Make HTTP calls to test the Webservice

# In[ ]:

import requests

url = webservice.scoring_uri
print('ACI Service: {} scoring URI is: {}'.format(service_name, url))
headers = {'Content-Type':'application/json'}

response = requests.post(url, json.dumps({'data': data1}), headers=headers)
print('Predictions for data1')
print(response.text)
response = requests.post(url, json.dumps({'data': data2}), headers=headers)
print('Predictions for data2')
print(response.text)

