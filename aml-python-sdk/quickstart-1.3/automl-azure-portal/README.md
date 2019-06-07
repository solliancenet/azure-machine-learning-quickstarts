# Prerequisites

- If an environment is provided to you. Use the workspace named: `quick-starts-ws-XXXXX`, where `XXXXX` is your unique identifier.

- If you are using your own Azure subscription. Create an Azure Machine Learning service workspace named: `quick-starts-ws`. See [Create an Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) for details on how to create the workspace.

- Download the training data file [nyc-taxi-sample-data.csv](https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv) on your local disk.

## Step 1: Navigate to Automated Machine Learning in Azure Portal

- Navigate to the machine learning workspace: `quick-starts-ws-XXXXX` or `quick-starts-ws`
- Select `Automated machine learning` in the left navigation bar
- Click on **Create Experiment**

<img src="./images/02_CreateExperiment.png" width="70%" height="70%" title="Create new experiment">

## Step 2: Create Experiment

- Provide an experiment name: `auto-ml-exp`
- Click on **Create a new compute**

<img src="./images/03_NewExperiment_1.png" width="70%" height="70%"  title="Provide experiment name and click on compute">

## Step 3: Create New Compute

- Provide compute name: `auto-ml-compute`
- Select your VM size
- Provide `Additional Settings`
- Click on **Create**
- Wait for Compute to be ready
- Click on **Next**

<img src="./images/04_CreateNewCompute.png" width="70%" height="70%" title="Create new compute">

## Step 4: Upload Training Data

- Click on **Upload**
- Upload `nyc-taxi-sample-data.csv` from your local disk

<img src="./images/05_UploadDataFile.png" width="70%" height="70%" title="Upload training data">

## Step 5: Review Training Data

- Click on **nyc-taxi-sample-data.csv**
- Review your training data. Scroll to right to observe the target column: `totalAmount`

<img src="./images/06_ReviewDataFile.png" width="70%" height="70%" title="Review training data">

## Step 6: Setup AutoML Experiment Basic Settings

- Select Prediction Task **Regression**
- Select Target column: **totalAmount**
- Click on **Advanced Settings**

<img src="./images/07_SetupExp_1.png" width="70%" height="70%" title="Setup AuoML experiment basic settings">

## Step 7: Setup AutoML Experiment Advanced Settings and Run Experiment

- Select Primary metric **spearman_correlation**
- Select Max number of iterations: **3**
- Select Number of Cross Validations: **5**
- Select Max concurrent iterations: **1**
- Scroll down and click on **Start** to run the experiment

<img src="./images/08_SetupExp_2.png" width="70%" height="70%" title="Setup AutoML Experiment Advanced Settings and Run Experiment">

## Step 8: Review Experiment Run Results

- The experiment will run for about *5-10 min*
- Observe the model performance for the primary metric for different iterations
- Scroll down to see a table view of different iterations
- Click on the iteration with the best **spearman_correlation** score. Note that the spearman_correlation measures the monotonic relationships between the predicted value and actual value. In this case, the model with spearman_correlation score closest to 1 is the best model.

<img src="./images/09_ReviewRunDetails_1.png" width="70%" height="70%" title="Review run details - graph view">
<img src="./images/010_ReviewRunDetails_2.png" width="70%" height="70%" title="Review run details - table view">

## Step 9: Review Best Model Predictions

- Review **Predicted Taxi Fare vs True Taxi Fare** for your best model

<img src="./images/011_ReviewPredictions.png" width="70%" height="70%" title="Review Best Model Predictions">

## Step 10: Review Best Model Metrics

- Scroll down to review various performance metrics for your best model

<img src="./images/012_ReviewMetrics.png" width="70%" height="70%" title="Review Best Model Metrics">

## Step 11: Deploy Best Model

- Return to `Run Details` screen
- Click on **Deploy Best Model** as shown
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

## Step 16: Clean-up

- We are done using the `Compute` resource.
- Navigate to the `Compute` tab and delete your compute target: `auto-ml-compute`

<img src="./images/013_DeleteCompute.png" width="70%" height="70%">

## Step 17: Proceed to Part 2

- [Automated ML with Azure ML Python SDK](../../quickstart-1.3#part-1-automated-ml-with-azure-ml-python-sdk)
