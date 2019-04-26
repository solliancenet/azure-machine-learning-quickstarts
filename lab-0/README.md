# Lab 0: Setting up your environment 

If a lab environmnet has not be provided for you, this lab provides the instructions to get started in your own Azure Subscription.

The following summarizes the lab requirements if you want to setup your own environment (for example, on your local machine). If this is your first time peforming these labs, it is highly recommended you follow the Quick Start instructions below rather that setup your own environment from scratch.

The labs have the following requirements:
- Azure subscription. You will need a valid and active Azure account to complete this Azure lab. If you do not have one, you can sign up for a [free trial](https://azure.microsoft.com/en-us/free/).
- One of the following environments:
    - Azure Notebooks
    - Visual Studio Code
    
The following sections describe the setup process for each environment.

# Quickstart: Azure Notebooks

The quickest way to get going with the labs is upload the files in Azure Notebooks. 

1. You can sign up for a free account with [Azure Notebooks](https://notebooks.azure.com/).

2. Create a new project in Azure Notebooks called “Aml-quickstarts”.

3. Create one folder for each lab in your project:  01-aml-pipelines, 02-aml-interpretability, 03-aml-deep-learning, 04-aml-onnx.

4. From starter-artifacts navigate to the [python-notebooks](../starter-artifacts/python-notebooks) link and load the files for each of the labs in their respective folders in your Azure Notebook project. You may have to download the files first to your local computer and then upload them to Azure notebooks. Also remember to maintain any folder structure within each of the labs. For example, the 04-aml-onnx lab has the notebook file “onnx-AML.ipynb” at the root and it has one subfolder called “model” that contains a pretrained model file.

5. To run a lab, you can start your project to run on “Free Compute”. Set Python 3.6 as your kernel for your notebooks. You can also configure Azure Notebooks to run on a Deep Learning Virtual Machine (DLVM). In the Azure Notebooks project, you can configure the notebooks to run on Direct Compute by providing the credentials for Azure Notebooks to connect to the DLVM compute target.

6. Next, follow the steps as outlined for each of labs.

# Quickstart: Visual Studio Code

1. Install [Visual Studio Code](https://code.visualstudio.com/docs/setup/setup-overview) on your machine.

2. Install the latest version of [Anaconda](https://www.anaconda.com/distribution/).

3. Setup a new conda environment for Azure Auto ML. The easiest way to do that is to download the automl_setup scripts and the YAML file for your machine from the following [GitHub repository](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/automated-machine-learning). After you run the script it will creates a new conda environment called azure_automl, and installs the necessary packages.

4. From starter-artifacts navigate to the [visual-studio-code](../starter-artifacts/visual-studio-code) and download the project files to your local computer. Also remember to maintain the folder structure within each of the labs. For example, the 04-aml-onnx lab has the python file “onnx-AML.py” at the root and it has one subfolder called “model” that contains a pretrained model file.

5. In VS code, when you first open the starting python file for a lab, use Select Interpreter command from the Command Palette (⇧⌘P) and select the azure_automl as your interpreter.

6. Next, follow the steps as outlined for each of labs.
