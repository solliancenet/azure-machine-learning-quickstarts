# Setting up your environment 

If a lab environment has not be provided for you, this lab provides the instructions to get started in your own Azure Subscription.

The labs have the following requirements:
- Azure subscription. You will need a valid and active Azure account to complete this Azure lab. If you do not have one, you can sign up for a [free trial](https://azure.microsoft.com/en-us/free/).

# Azure Quotas Required
The quickstarts depend on the capability to utilize a certain quantity of Azure resources, for which your Azure subscription will need to have sufficient quota available. 

The following are the specific quotas required, if your subscription does not meet the quota requirements in the region in which you will perform the quickstarts, you will need to request a quota increase thru Azure support:

Compute-VM
- Quota: Standard Dv2 Family vCPUs
- Provider: Microsoft.Compute
- SKU family: Dv2 Series
- Required Limit: 14

Compute-VM
- Quota: Total Regional vCPUs
- Provider: Microsoft.Compute
- SKU family: Dv2 Series
- Required Limit: 14


# Prerequisites

- Create an Azure resource group named: `QuickStarts`. See [Create Resource Groups](https://docs.microsoft.com/en-us/azure/azure-resource-manager/manage-resource-groups-portal) for details on how to create the resource group.

- Create an Azure Machine Learning service workspace named: `quick-starts-ws`. See [Create an Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) for details on how to create the workspace.

Next, we will upfront create two Azure Machine Learning Computes, one for running machine learning experiments and the other for deploying trained models. Creating computes can take upto 5 minutes, thus we can start the creation and move on to the quickstart to conserve time. 


# Create Azure Machine Learning Compute


### Step 1

Create a compute target in the workspace `quick-starts-ws` to run your Azure Machine Learning experiments.

Navigate to your workspace and select `Compute` from the `Assets` section and click on **Add Compute**:

<img src="./images/01.png" width="70%" height="70%" title="Click on Add Compute">


### Step 2

Create a Machine Learning Compute named `qs-compute` as follows:

1. Compute name: `qs-compute`
2. Compute type: `Machine Learning Compute`
3. Region: `Select your region`
4. Virtual machine size: `Standard_D2_v2`
5. Minimum number of nodes: `1`
6. Maximum number of nodes: `1`
7. Click on **Create**

<img src="./images/02.png" width="70%" height="70%" title="Create a Machine Learning Compute">


# Create Kubernetes Service Compute

Next, we will create a Kubernetes Service Compute to publish the trained model as web service.

Create Kubernetes Service compute as follows:

1. Compute name: `nyc-taxi`
2. Compute type: `Kubernetes Service`
3. Region: `Select your region`
4. Virtual machine size: `Standard_D3_v2`
5. Number of nodes: `3`
6. Click on **Create**

<img src="./images/03.png" width="70%" height="70%" title="Create Kubernetes Service Compute">


## Next, follow the steps as outlined for quickstart [Azure Machine Learning Visual Interface](../../aml-visual-interface/quickstart-2.1/README.md)
