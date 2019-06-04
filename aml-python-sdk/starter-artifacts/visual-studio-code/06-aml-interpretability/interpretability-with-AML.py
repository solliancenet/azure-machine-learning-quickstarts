# # Predicting NYC Taxi Fares And Model Explainer

# In this quickstart, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/). The data is enriched with holiday and weather data. We will use data transformations and the GradientBoostingRegressor algorithm from the scikit-learn library to train a regression model to predict taxi fares in New York City based on input features such as, number of passengers, trip distance, datetime, holiday information and weather information.
# 
# The primary goal of this quickstart is to explain the predictions made by our trained model with the various [Azure Model Interpretability](https://docs.microsoft.com/en-us/azure/machine-learning/service/machine-learning-interpretability-explainability) packages of the Azure Machine Learning Python SDK.

# ### Azure Machine Learning and Model Interpretability SDK-specific Imports

# In[ ]:

import os
import numpy as np
import pandas as pd
import joblib
import math
import pickle

import azureml
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model

from azureml.explain.model.tabular_explainer import TabularExplainer

from azureml.contrib.explain.model.tabular_explainer import TabularExplainer as TabularExplainer_contrib

print(azureml.core.VERSION)


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

experiment_name = "quick-starts-explain"


# ### Create and connect to an Azure Machine Learning Workspace
# 
# Run the following cell to create a new Azure Machine Learning **Workspace** and save the configuration to disk (next to the Jupyter notebook). 
# 
# **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.

# In[ ]:

working_directory = 'interpretability-with-AML'
os.makedirs(working_directory, exist_ok=True)
os.chdir(working_directory)


# In[ ]:

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)


# ### Train the Model

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

data_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
            'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
            'quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv')

df = pd.read_csv(data_url)
x_df = df.drop(['totalAmount'], axis=1)
y_df = df['totalAmount']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

categorical = ['normalizeHolidayName', 'isPaidTimeOff']
numerical = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 
             'day_of_month', 'month_num', 'snowDepth', 'precipTime', 'precipDepth', 'temperature']

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('regressor', GradientBoostingRegressor())])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
y_actual = y_test.values.flatten().tolist()
rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
print('The RMSE score on test data for GradientBoostingRegressor: ', rmse)


# ## Global Explanation Using TabularExplainer
# 
# **Global Model Explanation** is a holistic understanding of how the model makes decisions. It provides you with insights on what features are most important and their relative strengths in making model predictions.
# 
# [TabularExplainer](https://docs.microsoft.com/en-us/python/api/azureml-explain-model/azureml.explain.model.tabularexplainer?view=azure-ml-py) uses one of three explainers: TreeExplainer, DeepExplainer, or KernelExplainer, and is automatically selecting the most appropriate one for our use case. You can learn more about the underlying model explainers at [Azure Model Interpretability](https://docs.microsoft.com/en-us/azure/machine-learning/service/machine-learning-interpretability-explainability).
# 
# To initialize an explainer object, you need to pass your model and some training data to the explainer's constructor.
# 
# *Note that you can pass in your feature transformation pipeline to the explainer to receive explanations in terms of the raw features before the transformation (rather than engineered features).*

# In[ ]:

# "features" and "classes" fields are optional
tabular_explainer = TabularExplainer(clf.steps[-1][1], initialization_examples=X_train, 
                                     features=X_train.columns,  transformations=transformations)


# ### Get the global feature importance values
# 
# Run the below cell and observe the sorted global feature importance. You will note that `tripDistance` is the most important feature in predicting the taxi fares, followed by `hour_of_day`, and `day_of_week`.

# In[ ]:

# You can use the training data or the test data here
global_explanation = tabular_explainer.explain_global(X_test)
# Sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
dict(zip(sorted_global_importance_names, sorted_global_importance_values))


# ## Local Explanation
# 
# You can use the [TabularExplainer](https://docs.microsoft.com/en-us/python/api/azureml-explain-model/azureml.explain.model.tabularexplainer?view=azure-ml-py) for a single prediction. You can focus on a single instance and examine model prediction for this input, and explain why.
# 
# We will create two sample inputs to explain the individual predictions.
# 
# - **Data 1**
#  - 4 Passengers at 3:00PM, Friday July 5th, temperature 80F, travelling 10 miles
# 
# - **Data 2**
#  - 1 Passenger at 6:00AM, Monday January 20th, rainy, temperature 35F, travelling 5 miles

# In[ ]:

# Create the test dataset
columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', 
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']

data = [[1, 4, 10, 15, 4, 5, 7, 'None', False, 0, 0.0, 0.0, 80], 
        [1, 1, 5, 6, 0, 20, 1, 'Martin Luther King, Jr. Day', True, 0, 2.0, 3.0, 35]]

data_df = pd.DataFrame(data, columns = columns)


# In[ ]:

# explain the test data
local_explanation = tabular_explainer.explain_local(data_df)

# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
sorted_local_importance_values = local_explanation.get_ranked_local_values()

results = pd.DataFrame([sorted_local_importance_names[0][0:5], sorted_local_importance_values[0][0:5], 
                        sorted_local_importance_names[1][0:5], sorted_local_importance_values[1][0:5]], 
                       columns = ['1st', '2nd', '3rd', '4th', '5th'], 
                       index = ['Data 1', '', 'Data 2', ''])
print('Top 5 Local Feature Importance')
results


# As we saw from the Global Explanation that the **tripDistance** is the most important global feature. Other than `tripDistance`, the rest of the top 5 important features were different for the two samples.
# 
# - Data 1: Passenger count 4 and 3:00 PM on Friday were also important features in the prediction.
# - Data 2: The weather-related features (rainy, temperature 35F), day of the week (Monday) and month (January) were also important.

# ## Interpretability in Inference
# 
# In the next part, we will deploy the explainer along with the model to be used at scoring time to make predictions and provide local explanation.

# #### Create a Scoring Explainer
# 
# You have to use the **TabularExplainer** from the **azureml.contrib.explain.model.tabular_explainer** package to create the **Scoring Explainer** that can be deployed with the trained model.

# In[ ]:

tabular_explainer_contrib = TabularExplainer_contrib(clf.steps[-1][1], initialization_examples=X_train, 
                                     features=X_train.columns,  transformations=transformations)
scoring_explainer = tabular_explainer_contrib.create_scoring_explainer(X_test)


# ### Register the Trained Model and the Scoring Explainer with Azure Machine Learning Service
# 
# Run the next set of cells to register the two models with Azure Machine Learning Service.

# #### Start an Experiment Run

# In[ ]:

experiment = Experiment(ws, experiment_name)
run = experiment.start_logging()


# #### Save the two models to local disk

# In[ ]:

model_name = 'nyc-taxi-fare'
model_file_name = model_name + '.pkl'
os.makedirs('./outputs', exist_ok=True)
output_model_file_path = os.path.join('./outputs', model_file_name)
pickle.dump(clf, open(output_model_file_path,'wb'))

scoring_explainer_name = 'nyc-taxi-fare-explainer'
scoring_explainer_file_name = scoring_explainer_name + '.pkl'
scoring_explainer_file_path = os.path.join('./outputs', scoring_explainer_file_name)
scoring_explainer.save(scoring_explainer_file_path)


# #### Register the Models

# In[ ]:

# Upload and register this version of the model with Azure Machine Learning service
model_destination_path = os.path.join('outputs', model_name)
run.upload_file(model_destination_path, output_model_file_path) # destination, source
registered_model = run.register_model(model_name=model_name, model_path=model_destination_path)

scoring_explainer_destination_path = os.path.join('outputs', scoring_explainer_file_name)
run.upload_file(scoring_explainer_destination_path, scoring_explainer_file_path) # destination, source
registered_scoring_explainer = run.register_model(model_name=scoring_explainer_name, 
                                                  model_path=scoring_explainer_destination_path)

print("Model registered: {} \nModel Version: {}".format(registered_model.name, registered_model.version))
print("Explainer Model registered: {} \nExplainer Model Version: {}".format(registered_scoring_explainer.name, 
                                                                            registered_scoring_explainer.version))


# #### Complete the Experiment Run

# In[ ]:

run.complete()


# ### Create the Scoring Script

# In[ ]:

get_ipython().run_cell_magic('writefile', 'score.py', "import json\nimport numpy as np\nimport pandas as pd\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom azureml.core.model import Model\n\ncolumns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', \n           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', \n           'precipDepth', 'temperature']\n\ndef init():\n\n    global original_model\n    global scoring_explainer\n\n    # Retrieve the path to the model file using the model name\n    # Assume original model is named original_prediction_model\n    original_model_path = Model.get_model_path('nyc-taxi-fare')\n    scoring_explainer_path = Model.get_model_path('nyc-taxi-fare-explainer')\n\n    original_model = joblib.load(original_model_path)\n    scoring_explainer = joblib.load(scoring_explainer_path)\n\ndef run(input_json):\n    # Get predictions and explanations for each data point\n    inputs = json.loads(input_json)\n    data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)\n    # Make prediction\n    predictions = original_model.predict(data_df)\n    # Retrieve model explanations\n    local_importance_values = scoring_explainer.explain(data_df)\n    # You can return any data type as long as it is JSON-serializable\n    return {'predictions': predictions.tolist(), 'local_importance_values': local_importance_values}")


# ### Package Model
# 
# Run the next two cells to create the deployment **Image**
# 
# *WARNING: to install, g++ needs to be available on the Docker image and is not by default. Thus, we will create a custom dockerfile with g++ installed.*

# In[ ]:

get_ipython().run_cell_magic('writefile', 'dockerfile', 'RUN apt-get update && apt-get install -y g++')


# In[ ]:

# create a Conda dependencies environment file
print("Creating conda dependencies file locally...")
from azureml.core.conda_dependencies import CondaDependencies 
conda_packages = ['numpy', 'pandas', 'scikit-learn']
pip_packages = ['azureml-sdk', 'azureml-explain-model', 'azureml-contrib-explain-model', 'sklearn_pandas']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

conda_file = 'sklearn_dependencies.yml'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

runtime = 'python'

# create container image configuration
print("Creating container image configuration...")
from azureml.core.image import ContainerImage
image_config = ContainerImage.image_configuration(execution_script = 'score.py', 
                                                  docker_file = 'dockerfile', 
                                                  runtime = runtime, 
                                                  conda_file = conda_file)

# create the image
image_name = 'nyc-taxi-fare-image'

from azureml.core import Image
image = Image.create(name=image_name, models=[registered_model, registered_scoring_explainer], 
                     image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)


# ### Deploy Model to Azure Container Instance (ACI) as a Web Service

# In[ ]:

from azureml.core.webservice import AciWebservice, Webservice

aci_name = 'sklearn-aci-cluster01'

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name': aci_name}, 
    description = 'NYC Taxi Fare Predictor Web Service')

service_name = 'nyc-taxi-sklearn-service'

aci_service = Webservice.deploy_from_image(deployment_config=aci_config, 
                                           image=image, 
                                           name=service_name, 
                                           workspace=ws)

aci_service.wait_for_deployment(show_output=True)

# ### Test Deployment
# 
# Observe that the **Scoring Service** return both prediction and local importance values.

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


# ### Cleanup

# In[ ]:

# Change back to the root directory
os.chdir('../')

