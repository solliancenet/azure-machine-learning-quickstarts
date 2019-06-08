## Before you begin

Download the training data file [nyc-taxi-sample-data.csv](https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv) on your local disk.

## Step 1: Navigate to Visual Interface in Azure Portal

- Navigate to the machine learning workspace: `quick-starts-ws`
- Select `Visual interface` in the left navigation bar
- Click on **Launch visual interface**

<img src="./images/01.png" width="70%" height="70%" title="Launch visual interface" border="15">

## Step 2: Upload Training Dataset

- Click on **+ New**
- Select **Datasets**
- Click on **Upload from Local File**
- Navigate to `nyc-taxi-sample-data.csv` on your local disk and click **Ok**

<img src="./images/02_1.png" width="70%" height="70%" title="Click on + New" border="15">

<img src="./images/02_2.png" width="70%" height="70%" title="Click on Upload from Local File" border="15">

<img src="./images/02_3.png" width="70%" height="70%" title="Click on Ok" border="15">

## Step 3: Create New Blank Experiment and Select your Dataset

- Click on **+ New**
- Select **+ Blank Experiment**
- Expand **My Datasets** in the left navigation
- Drag and drop your dataset `nyc-taxi-sample-data.csv` on to the canvas

<img src="./images/03_1.png" width="70%" height="70%" title="Click on + Blank Experiment" border="15">

<img src="./images/03_2.png" width="70%" height="70%" title="Expand My Datasets" border="15">

<img src="./images/03_3.png" width="70%" height="70%" title="Drag and Drop My Dataset" border="15">

## Step 4: Transform your data

- Open **Data Transformation** section in the left panel
- Drag and drop **Select Columns in Dataset** module on to the canvas
- Select the added `Select Columns in Dataset` module
- Click on **Edit columns** in the right panel
- Exclude the column name `vendorID` from your dataset
- Add **Clean Missing Data** module
- Sequentially connect the three modules
- Select the added  `Clean Missing Data` module
- Select `Replace with mean` as `Cleaning mode` in the right panel
- Click on **Edit columns** in the right panel
- Exclude the two non-numeric data columns `normalizedHolidayName` and `isPaidTimeOff`
- Split the data for training and evaluation using the **Split Data** module

<img src="./images/04_1.png" width="70%" height="70%" title="Select Columns in Dataset Module" border="15">

<img src="./images/04_2.png" width="70%" height="70%" title="Exclude vendorID column" border="15">

<img src="./images/04_3.png" width="70%" height="70%" title="Clean Missing Data Module" border="15">

<img src="./images/04_4.png" width="70%" height="70%" title="Exclude non-numeric columns" border="15">

<img src="./images/04_4.png" width="70%" height="70%" title="Split Data Module" border="15">

## Step 5: Initialize your Regression Model

- Open **Machine Learning -> Initialize Model -> Regression** section in the left panel
- Add the **Boosted Decision Tree Regression** module on to the canvas
- Configure your model parameters

<img src="./images/05_1.png" width="70%" height="70%" title="Boosted Decision Tree Regression Module"  border="15">

## Step 6: Setup Train Model Module

- Open **Machine Learning -> Train** section in the left panel
- Add **Train Model** module on to the canvas
- Click on **Edit columns** in the right panel to setup your `Label or Target column`
- Select `totalAmount` as your target column

<img src="./images/06_1.png" width="70%" height="70%" title="Train Model Module"  border="15">

<img src="./images/06_2.png" width="70%" height="70%" title="Setup totalAmount as target column"  border="15">

## Step 7: Setup Score Model and Evaluate Model Modules

- Open **Machine Learning -> Score** section in the left panel
- Add **Score Model** module on to the canvas
- Complete the model training and scoring connections
- Note that `Split Data` module will feed data for both model training and model scoring
- Open **Machine Learning -> Evaluate** section in the left panel
- Add **Evaluate Model** module on to the canvas
- Connect the `Score Model` module to `Evaluate Model` module

<img src="./images/07_1.png" width="70%" height="70%" title="Score Model Module"  border="15">

<img src="./images/07_2.png" width="70%" height="70%" title="Evaluate Model Module"  border="15">

## Step 8: Run the Experiment

- Click on **Run**
- Note that you can create a new **Compute Target** directly from **Visual Interface**
- Select the existing compute target we created upfront in `quickstart-2.0`
- Click on **Run**
- The experiment will run for about 8-10 minutes 

<img src="./images/08_1.png" width="70%" height="70%" title="Click on Run Experiment"  border="15">

<img src="./images/08_2.png" width="70%" height="70%" title="Select Compute and Run Experiment"  border="15">

## Step 9: Visualize the Model Predictions

- Wait for model training to be complete
- Right click on **Score Model** module and select **Scored dataset -> Visualize**
- Compare the predicted `predicted taxi fares` to the `target taxi fares`
- You can also observe the predicted value distribution when you select the `Scored Labels` column 
 
<img src="./images/09_1.png" width="70%" height="70%" title="Visualize the Model Predictions" border="15">

<img src="./images/09_2.png" width="70%" height="70%" title="Compare prediction to actual values" border="15">

## Step 10: Visualize the Evaluation Results

- Right click on **Evaluate Model** module and select **Evaluation results -> Visualize**
- Observe the Model Evaluation Metrics such as **Root Mean Squared Error**

<img src="./images/10_1.png" width="70%" height="70%" title="Visualize the Model Evaluation" border="15">

<img src="./images/10_2.png" width="70%" height="70%" title="Model Evaluation Metrics" border="15">

## Step 11: Experiment Run History

- Click on **Run History**
- Go back to the Experiment by clicking on the `Editable` version of the Run History

<img src="./images/11_1.png" width="70%" height="70%" title="Go to Run History" border="15">

<img src="./images/11_2.png" width="70%" title="Go back to the experiment">

## Step 12: Save the Trained Model

- Right click on the `Train Model` module and select **Trained model -> Save as Trained Model**
- Save the Trained Model

<img src="./images/12_1.png" width="70%" height="70%" title="Right click on the Train Model module" border="15">

<img src="./images/12_2.png" width="70%" height="70%" title="Save the Trained Model" border="15">

## Step 13: Run Predictive Experiment

- Observe the saved model appears in the `Trained Models` section
- Click on **Create Predictive Experiment**
- Click on **Run**
- Select the existing compute target to run the experiment
- It will take about 4-5 minutes to run the predictive experiment

<img src="./images/13_1.png" width="70%" height="70%" title="Create Predictive Experiment" border="15">

<img src="./images/13_2.png" width="70%" height="70%" title="Click on Run" border="15">

## Step 14: Deploy Web Service

- Click on **Deploy Web Service**
- Select the existing Kubernetes Service compute, and click **Deploy**

<img src="./images/14_1.png" width="70%" height="70%" title="Deploy Web Service" border="15">

<img src="./images/14_2.png" width="70%" height="70%" title="Deploy Web Service" border="15">

## Step 15: Test the Deployed Web Service

- Navigate to the Web Services section of the Visual Interface, and open the web service
- Fill in the data to predict with your web service, and click **Test**
- Observed the **Raw** test results and predicted output **Scored Labels**

<img src="./images/15_1.png" width="70%" height="70%" title="Open the Web Service" border="15">

<img src="./images/15_2.png" width="70%" height="70%" title="Test the Web Service" border="15">

## Step 16: Review how to Consume the Deployed Web Service

- Navigate to the **Consume** section of the Web Service
- Observe the provided sample code in C#, Python, and R to consume the Web Service

<img src="./images/16_1.png" width="70%" height="70%" title="Consume the Deployed Web Service" border="15">

## Step 17: Cleanup Resources

- Go to Azure Portal and navigate to the Deployments section of the workspace
- **Delete** the deployed Web Service
- Navigate to **Compute** section
- **Delete** both the compute targets created for this quickstart - one at a time

<img src="./images/17_1.png" width="70%" height="70%" title="Delete the deployed Web Service" border="15">

<img src="./images/17_2.png" width="70%" height="70%" title="Delete the Compute Targets" border="15">
