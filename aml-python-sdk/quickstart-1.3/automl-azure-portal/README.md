# Prerequisites

- If an environment is provided to you. Use the workspace named: `quick-starts-ws-XXXXX`, where `XXXXX` is your unique identifier.

- If you are using your own Azure subscription. Create an Azure Machine Learning service workspace named: `quick-starts-ws`. See [Create an Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) for details on how to create the workspace.

- Download the training data file [nyc-taxi-sample-data.csv](https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv) on your local disk.

# Exercise 1: Setup New Automated Machine Learning Experiment

## Task 1: Create New Automated Machine Learning Experiment

1. In Azure Portal, open the machine learning workspace: `quick-starts-ws-XXXXX` or `quick-starts-ws`
2. Select `Automated machine learning` in the left navigation bar
3. Select **Create Experiment** from the content section
4. This will open a `Create a new automated machine learning experiment` page

  <img src="./images/02_CreateExperiment.png" width="70%" height="70%" title="Create new experiment">

5. Provide an experiment name: `auto-ml-exp`
6. Select **Create a new compute**

  <img src="./images/03_NewExperiment_1.png" width="70%" height="70%"  title="Provide experiment name and click on compute">

## Task 2: Create New Compute

1. Provide compute name: `auto-ml-compute`
2. Select your VM size: `Standard_DS3_v2`
3. Provide `Additional Settings`

   a. Minimum number of nodes: 1
   
   b. Maximum number of nodes: 1
   
4. Select **Create**
5. Wait for compute to be ready. This may take 2-3 minutes.
6. Select **Next**

  <img src="./images/04_CreateNewCompute.png" width="70%" height="70%" title="Create new compute">

# Exercise 2: Upload and Review Training Data

## Task 1: Upload Training Data

- Select **Upload**
- Upload `nyc-taxi-sample-data.csv` from your local disk

  <img src="./images/05_UploadDataFile.png" width="70%" height="70%" title="Upload training data">

## Task 2: Review Training Data

- Select **nyc-taxi-sample-data.csv** from the list

  <img src="./images/06_ReviewDataFile.png" width="70%" height="70%" title="Select training data">

- Review your training data. Scroll to right to observe the target column: `totalAmount`

  <img src="./images/061_ReviewDataFile.png" width="70%" height="70%" title="Review training data">

# Exercise 3: Setup Experiment Settings

## Task 1: Basic Settings

1. Select Prediction Task: **Regression**
2. Select Target column: **totalAmount**
3. Open **Advanced Settings**

  <img src="./images/07_SetupExp_1.png" width="70%" height="70%" title="Setup experiment basic settings">

## Task 2: Advanced Settings

1. Select Primary metric **spearman_correlation**
2. Select Max number of iterations: **3**
3. Select Number of Cross Validations: **5**
4. Select Max concurrent iterations: **1**

  <img src="./images/08_SetupExp_2.png" width="70%" height="70%" title="Setup experiment advanced settings">

# Exercise 4: Start and Monitor Experiment

## Task 1: Start Experiment

1. Scroll down and select **Start** to run the experiment

  <img src="./images/09_StartExp.png" width="70%" height="70%" title="Start Experiment">

## Task 2: Monitor Experiment

1. The experiment will run for about *5-10 min*
2. In the **Run Details** screen, observe the performance of the various models for the primary metric: **spearman_correlation**

  <img src="./images/09_ReviewRunDetails_1.png" width="70%" height="70%" title="Review run details - graph view">
  
3. Scroll down to see a table view of different iterations
4. Wait for the experiment to complete

# Exercise 5: Review Best Model's Performance

## Task 1: Review Best Model Predictions

1. From the table view, select the iteration with the best **spearman_correlation** score. Note that the spearman_correlation measures the monotonic relationships between the predicted value and actual value. In this case, the model with spearman_correlation score closest to 1 is the best model.

  <img src="./images/010_ReviewRunDetails_2.png" width="70%" height="70%" title="Review run details - table view">
  
2. Review **Predicted Taxi Fare vs True Taxi Fare** for your model

  <img src="./images/011_ReviewPredictions.png" width="70%" height="70%" title="Review Best Model Predictions">

## Task 2: Review Best Model Metrics

1. Scroll down to review various performance metrics for your model

  <img src="./images/012_ReviewMetrics.png" width="70%" height="70%" title="Review Best Model Metrics">

# Exercise 6: Deploy Best Model

- Return to `Run Details` screen
- Select **Deploy Best Model** as shown
- Note that deployment consists of four steps: (1) *Register Best Model*, (2) Download *Scoring and Environment Script files*, (3) Create *Deployment Image* using the downloaded script files, and (4) Deploy *Scoring Web Service* using the created image.

<img src="./images/014_DeployBestModel.png" width="70%" height="70%" title="Deploy Best Model">

## Step 12: Register Best Model

- Click on **Register Model** link.

<img src="./images/015_RegisterModel.png" width="70%" height="70%" title="Register Best Model">

## Step 13: Download the Script Files

- Click on **Download Scoring Script** link. This will download `scoring.py` file to your local disk.
- Click on **Download Environment Script** link. this will download `condaEnv.yml` file to your local disk.

<img src="./images/016_DownloadScripts.png" width="70%" height="70%" title="Download the Script Files">

## Step 14: Create the Deployment Image

- Navigate to the `Models` section in your Azure Portal Workspace.
- Select the `Registered Model`
- Click on **Create Image**
- On the `Create an Image` page provide the information as shown below. Note that you should upload the corresponding script files downloaded in step # 13.
- Click on **Create** button.
- Creating an image can take upto 5-10 minutes. Wait for the image to be created before proceeding.

<img src="./images/017_CreateImage_1.png" width="70%" height="70%" title="Click on Create Image">

<img src="./images/017_CreateImage_2.png" width="70%" height="70%" title="Create an Image Information Page">

## Step 15: Deploy the Scoring Web Service

- Navigate to the `Images` section in your Azure Portal Workspace.
- Select the image created in step #14
- Confirm that the image creation succeeded.
- Click on **Create Deployment**
- On the `Create Deployment` page provide the information as shown below. We will deploy the scoring web service on an Azure Container Instance (ACI).
- Click on **Create** button.
- Creating a scoring web service can take upto 5 minutes.
- Navigate to the `Deployments` section in your Azure Portal Workspace. Wait for the service to be ready before proceeding.

<img src="./images/018_CreateDeployment_1.png" width="70%" height="70%" title="Click on Create Deployment">

<img src="./images/018_CreateDeployment_2.png" width="70%" height="70%" title="Create Deployment Information Page">

<img src="./images/018_CreateDeployment_3.png" width="70%" height="70%" title="Create Deployment Information Page">

## Step 16: Challenge Experiment
In the current experiment, the pipeline of `MaxAbsScaler, RandomForest` gave us the best performing model with the spearman correlation score of: **0.934**. Can you expand the number of iterations for the Automated Machine Learning experiment to see if we can find a better performing model? Note that `Number of iterations` parameter is defined as follows: *In each iteration, a new machine learning model is trained with your data. This is the primary value that affects total run time.*

## Step 17: Clean-up

- We are done using the `Compute` resource.
- Navigate to the `Compute` tab and delete your compute target: `auto-ml-compute`

<img src="./images/013_DeleteCompute.png" width="70%" height="70%">

## Step 18: Proceed to Part 2

- [Automated ML with Azure ML Python SDK](../../quickstart-1.3#part-2-automated-ml-with-azure-ml-python-sdk)
