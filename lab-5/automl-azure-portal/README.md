# Prerequisites

- Create an Azure Machine Learning service workspace named: `auto-ml-demo`. See [Create an Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) for details on how to create the workspace.

## Step 1: Navigate to Automated Machine Learning in Azure Portal

- Navigate to the machine learning workspace: `auto-ml-demo`
- Select `Automated machine learning` in the left navigation bar
- Click on **Create Experiment**

<img src="./images/02_CreateExperiment.png" width="70%" height="70%">

## Step 2: Create Experiment

- Provide an experiment name: `auto-ml-exp`
- Click on **Create a new compute**

<img src="./images/03_NewExperiment_1.png" width="70%" height="70%">

## Step 3: Create New Compute

- Provide compute name: `auto-ml-compute`
- Select your VM size
- Provide `Additional Settings`
- Click on **Create**
- Wait for Compute to be ready
- Click on **Next**

<img src="./images/04_CreateNewCompute.png" width="70%" height="70%">

## Step 4: Upload Training Data

- Click on **Upload**
- Upload `nyc-taxi-sample-data.csv` from your local disk

<img src="./images/05_UploadDataFile.png" width="70%" height="70%">

## Step 5: Review Training Data

- Click on **nyc-taxi-sample-data.csv**
- Review your training data. Scroll to right to observe the target column: `totalAmount`

<img src="./images/06_ReviewDataFile.png" width="70%" height="70%">

## Step 6: Setup AutoML Experiment Basic Settings

- Select Prediction Task **Regression**
- Select Target column: **totalAmount**
- Click on **Advanced Settings**

<img src="./images/07_SetupExp_1.png" width="70%" height="70%">

## Step 7: Setup AutoML Experiment Advanced Settings and Run Experiment

- Select Primary metric **spearman_correlation**
- Select Max number of iterations: **3**
- Select Number of Cross Validations: **5**
- Select Max concurrent iterations: **1**
- Scroll down and click on **Start** to run the experiment

<img src="./images/08_SetupExp_2.png" width="70%" height="70%">

## Step 8: Review Experiment Run Results

## Step 9: Review Best Model Predictions

## Step 10: Review Best Model Metrics

## Step 11: Clean-up
