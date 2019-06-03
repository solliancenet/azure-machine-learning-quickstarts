# Prerequisites

- If an environment is provided to you. Use the workspace named: `quick-starts-ws-XXXXX`, where `XXXXX` is your unique identifier.

- If you are using your own Azure subscription. Create an Azure Machine Learning service workspace named: `quick-starts-ws`. See [Create an Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) for details on how to create the workspace.

- Download the training data file [nyc-taxi-sample-data.csv](https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv) on your local disk.

## Step 1: Navigate to Automated Machine Learning in Azure Portal

- Navigate to the machine learning workspace: `quick-starts-ws-XXXXX` or `quick-starts-ws`
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

- The experiment will run for about *5-10 min*
- Observe the model performance for the primary metric for different iterations
- Scroll down to see a table view of different iterations
- Click on the iteration with the best **spearman_correlation** score

<img src="./images/09_ReviewRunDetails_1.png" width="70%" height="70%">
<img src="./images/010_ReviewRunDetails_2.png" width="70%" height="70%">

## Step 9: Review Best Model Predictions

- Review **Predicted Taxi Fare vs True Taxi Fare** for your best model

<img src="./images/011_ReviewPredictions.png" width="70%" height="70%">

## Step 10: Review Best Model Metrics

- Scroll down to review various performance metrics for your best model

<img src="./images/012_ReviewMetrics.png" width="70%" height="70%">

## Step 11: Clean-up

- Navigate to the `Compute` tab and delete your compute target: `auto-ml-compute`

<img src="./images/013_DeleteCompute.png" width="70%" height="70%">

## Step 12: Proceed to Part 2

- [Automated ML with Azure ML Python SDK](../../lab-5)
