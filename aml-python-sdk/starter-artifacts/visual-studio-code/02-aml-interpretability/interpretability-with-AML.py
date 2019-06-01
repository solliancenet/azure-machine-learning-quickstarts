'''
Predicting Car Battery Failure And Model Explainer
Your goal in this notebook is to **predict how much time a car battery has left until it is expected to fail and use the automated machine learning model explainer to retrieve important features for predictions**. You are provided training data that includes telemetry from different vehicles, as well as the expected battery life that remains. From this you will train a model that given just the vehicle telemetry predicts the expected battery life. You will use automlexplainer module to retrieve feature importance for all iterations and then explain the model with previously unseen test data.
 
You will use Azure Machine Learning SDK to **locally** train a **set** of models using **Automated Machine Learning**, evaluate performance of each model and pick the best performing model to retrive important features.

Setup
To begin, you will need to provide the following information about your Azure Subscription. 

In the following cell, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments (*these values can be acquired from the Azure Portal*).
 
**Be sure to replace XXXXX in the values below with your unique identifier.**
 
To get these values, do the following:
1. Navigate to the Azure Portal and login with the credentials provided.
2. From the left hand menu, under Favorites, select `Resource Groups`.
3. In the list, select the resource group with the name similar to `XXXXX`.
4. From the Overview tab, capture the desired values.
 
Execute the following cell by clicking the Run Cell below.
'''

# In[ ]:
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "xxx-xxx-xxx"

#Provide values for the existing Resource Group 
resource_group = "Interpretability-Lablet"

#Provide the Workspace Name and Azure Region of the Azure Machine Learning Workspace
workspace_name = "interpretability-workspace"
workspace_region = "eastus"

experiment_name = 'interpretability'
project_folder = './interpretability'

# this is the URL to the CSV file containing the training data
data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/training-formatted.csv"

# this is the URL to the CSV file containing a small set of test data
test_data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/fleet-formatted.csv"

'''
Import required packages

The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:
- Creating a Workspace that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
- Creating Experiments in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
- Creating Compute resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU. 
- Using Automated Machine Learning (AutoML) to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify. 
- Using *automlexplainer* module to retrive model explanation form existing run and explain model on new datasets.
- Packaging a Docker Image that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
- Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the Web Service.
 
In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following cell to do so:
'''

# In[ ]:
import logging
import os
import random
import re

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.model import Model
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.train.automl.automlexplainer import retrieve_model_explanation
from azureml.train.automl.automlexplainer import explain_model

'''
Create and connect to an Azure Machine Learning Workspace

Run the following cell to create a new Azure Machine Learning Workspace and save the configuration to disk (next to the Jupyter notebook). 
 
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
Create a Workspace Experiment
Notice in the first line of the cell below, we can re-load the config we saved previously and then display a summary of the environment.
'''

# In[ ]:
ws = Workspace.from_config()

# Display a summary of the current environment 
output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
pd.set_option('display.max_colwidth', -1)
pd.DataFrame(data=output, index=['']).T

'''
# Next, create a new Experiment.
'''
# In[ ]:
experiment = Experiment(ws, experiment_name)

'''
Get and explore the Vehicle Telemetry Data

Run the following cell to download and examine the vehicle telemetry data. The model you will build will try to predict how many days until the battery has a freeze event. Which features (columns) do you think will be useful?
'''

# In[ ]:
data = pd.read_csv(data_url)
data.head()

'''
Locally train multiple models using Auto ML

In the following cells, you will use Auto ML to train multiple models, evaluate the performance and allow you to retrieve the best model that was trained.

Create the data loading script for local compute

The Azure Machine Learning needs to know how to get the data to train against. You can package this logic in a script that will be executed by the AML when it starts executing the training.
 
Run the following cells to locally create the **get_data.py** script that will be used to train the model. 
 
Observe that the get_data method returns the features (`X`) and the labels (`Y`) in an object. The method also returns the validation set, (`X_valid`) and (`y_valid`), to use for model evaluation. This structure is expected later when you will configure Auto ML.
'''

# In[ ]:
# create project folder
if not os.path.exists(project_folder):
    os.makedirs(project_folder)

# In[ ]:
get_ipython().run_cell_magic('writefile', '$project_folder/get_data.py', '\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\ndef get_data():\n    \n    data_url = "https://databricksdemostore.blob.core.windows.net/data/connected-car/training-formatted.csv"\n    data = pd.read_csv(data_url)\n    \n    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:73], \n                                                        data.iloc[:,0].values.flatten(),\n                                                        random_state=0)\n\n    return { "X": X_train, "y": y_train, "X_valid":  X_test, "y_valid": y_test}')

'''
Instantiate an Automated ML Config

Run the following cell to configure the Auto ML run. In short what you are configuring here is the training of a regressor model that will attempt to predict the value of the first feature (`Survival_in_days`) based on all the other features in the data set. The run is configured to try at most 3 iterations where no iteration can run longer that 2 minutes. 
 
Additionally, the data will be automatically pre-processed in different ways as a part of the automated model training (as indicated by the `preprocess` attribute having a value of `True`). This is a very powerful feature of Auto ML as it tries many best practices approaches for you, and saves you a lot of time and effort in the process. When you look at important features from the model explainer you will observe the preprocessed features the Auto ML created for the best run.

The goal of Auto ML in this case is to find the best models that result, as measure by the normalized root mean squared error metric (as indicated by the `primary_metric` attribute). The error is basically a measure of what the model predicts versus what was provided as the "answer" in the training data. In short, AutoML will try to get the error as low as possible when trying its combination of approaches.  
 
The local path to the script you created to retrieve the data is supplied to the AutoMLConfig, ensuring the file is made available during training.
 
In general, the AutoMLConfig is very flexible, allowing you to specify all of the following:
- Task type (classification, regression, forecasting)
- Number of algorithm iterations and maximum time per iteration
- Accuracy metric to optimize
- Algorithms to blacklist (skip)/whitelist (include)
- Number of cross-validations
- Compute targets
- Training data
 
Run the following cell to create the configuration.
'''

# In[ ]:
automl_config = AutoMLConfig(task = 'regression',
                             iterations = 3,
                             iteration_timeout_minutes = 2, 
                             max_cores_per_iteration = 10,
                             preprocess= True,
                             primary_metric='normalized_root_mean_squared_error',
                             model_explainability=True,
                             debug_log = 'automl.log',
                             verbosity = logging.DEBUG,
                             data_script = project_folder + "/get_data.py",
                             path = project_folder)


#Run our Auto ML Experiment on Local Machine

# In[ ]:
local_run = experiment.submit(automl_config, show_output=True)
local_run.wait_for_completion(show_output = True)

'''
List the Experiments from your Workspace
Using the Azure Machine Learning SDK, you can retrieve any of the experiments in your Workspace and drill into the details of any runs the experiment contains. Run the following cell to explore the number of runs by experiment name.
'''

# In[ ]:
ws = Workspace.from_config()
experiment_list = Experiment.list(workspace=ws)

summary_df = pd.DataFrame(index = ['No of Runs'])
pattern = re.compile('^AutoML_[^_]*$')
for experiment in experiment_list:
    all_runs = list(experiment.get_runs())
    automl_runs = []
    for run in all_runs:
        if(pattern.match(run.id)):
            automl_runs.append(run)    
    summary_df[experiment.name] = [len(automl_runs)]
    
pd.set_option('display.max_colwidth', -1)
summary_df.T

'''
List the Automated ML Runs for the Experiment
Similarly, you can view all of the runs that ran supporting Auto ML:
'''

# In[ ]:
proj = ws.experiments[experiment_name]
summary_df = pd.DataFrame(index = ['Type', 'Status', 'Primary Metric', 'Iterations', 'Compute', 'Name'])
pattern = re.compile('^AutoML_[^_]*$')
all_runs = list(proj.get_runs(properties={'azureml.runsource': 'automl'}))
for run in all_runs:
    if(pattern.match(run.id)):
        properties = run.get_properties()
        tags = run.get_tags()
        amlsettings = eval(properties['RawAMLSettingsString'])
        if 'iterations' in tags:
            iterations = tags['iterations']
        else:
            iterations = properties['num_iterations']
        summary_df[run.id] = [amlsettings['task_type'], run.get_details()['status'], properties['primary_metric'], iterations, properties['target'], amlsettings['name']]
    
from IPython.display import HTML
projname_html = HTML("<h3>{}</h3>".format(proj.name))

from IPython.display import display
display(projname_html)
display(summary_df.T)

'''
Get the best run and the trained model
At this point you have multiple runs, each with a different trained models. How can you get the model that performed the best? Run the following cells to learn how.
'''

# In[ ]:
best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)

'''
Best Model's explanation

Retrieve the explanation from the best_run. And explanation information includes:
 
1. shap_values: The explanation information generated by shap lib
2. expected_values: The expected value of the model applied to set of X_train data.
3. overall_summary: The model level feature importance values sorted in descending order
4. overall_imp: The feature names sorted in the same order as in overall_summary
5. per_class_summary: The class level feature importance values sorted in descending order. Only available for the classification case
6. per_class_imp: The feature names sorted in the same order as in per_class_summary. Only available for the classification case
 
In the case we are only interested in looking at **overall_summary** and **overall_imp** to observe the top 10 features by their importance
'''

# In[ ]:
_, _, overall_summary, overall_imp, _, _ = retrieve_model_explanation(best_run)

important_features = pd.DataFrame(data = {'feature': overall_imp[0:10], 'value': overall_summary[0:10]},
                                 index = [i for i in range(1, 11)])
print('Top 10 important features')
print(important_features)

'''
Make predictions on the test data
Run the following cell to download and examine new vehicle telemetry dataset that was not used in model training or model evaluation.
'''

# In[ ]:
test_data = pd.read_csv(test_data_url)
test_data = test_data.drop(columns=["Car_ID", "Battery_Age"])
test_data.rename(columns={'Twelve_hourly_temperature_forecast_for_next_31_days_reversed': 
                          'Twelve_hourly_temperature_history_for_last_31_days_before_death_last_recording_first'}, 
                 inplace=True)
test_data

# Use the fitted model to make **battery life** predictions

# In[ ]:
predictions = fitted_model.predict(test_data)
predictions = pd.DataFrame(predictions, columns=['Battery Life Predictions'])
print(predictions)

'''
Model Explanation on Test Data

Explain the model with the new test data - dataset that was not used in model training or model evaluation. The **explain_model** call takes in two parameters
1. A matrix of feature vector examples for initializing the explainer.
2. A matrix of feature vector examples on which to explain the model's output.
 
To initialize the explainer we will use the original training set data and use the test data to retrive the model explanation. 
 
In this example, let's observe the top 10 features by their importance when predicting one of the test samples. We select the row 7 (index 6) of the test data that predicted the highest battery life of 2195.
'''

# In[ ]:
# select the row 7 (index 6) from the test data
row_7 = test_data.iloc[[6]]

# Initialize with the original test features data.iloc[:,1:73] and return the model explanation for row_7
_, _, overall_summary, overall_imp, _, _ = explain_model(fitted_model, data.iloc[:,1:73], row_7)

important_features_test = pd.DataFrame(data = {'feature': overall_imp[0:10], 'value': overall_summary[0:10]},
                                 index = [i for i in range(1, 11)])
print('Top 10 important features for row 7')
print(important_features_test)

'''
Based on the explain_model output, we can see that the following orginal features had the greatest influence in the row 7 prediction:
 
1. Province
2. Battery_Rated_Cycles
3. Trip_Length_Mean
4. Car_Has_EcoStart
5. Alternator_Efficiency
 
Let's look the origial important feature values and the corresponding model predictions.
'''

# In[ ]:
print(pd.concat([test_data[['Province', 'Battery_Rated_Cycles', 
'Trip_Length_Mean', 'Car_Has_EcoStart', 
'Alternator_Efficiency']], predictions], axis=1))

'''
For row 7 / index 6, The Battery_Rated_Cycles value 300 was highest amongst the test data, whereas, the other values such as Trip_Length_Mean 14.07 and Alternator_Efficiency 0.83 were close to the respective mean values of the test set. Thus, it appears that for row 7 / index 6, the two most differentiating features were the **Province: Marseille** and **Battery_Rated_Cycles** and these features were also identified as the top two important features.
'''