# Technology overview

## Automated Machine Learning with Azure Machine Learning
Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.

Configuration options available in automated machine learning:

- Select your experiment type: Classification, Regression or Time Series Forecasting
- Data source, formats, and fetch data
- Choose your compute target: local or remote
- Automated machine learning experiment settings
- Run an automated machine learning experiment
- Explore model metrics
- Register and deploy model

You can create and run automated machine learning experiments in code using the [Azure ML Python SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train) or if you prefer a no code experience, you can also Create your automated machine learning experiments in the [Azure portal](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-create-portal-experiments).

# Lab Overview
This lab is divided into two parts. In the first part, you learn how to create, run, and explore automated machine learning experiments in the Azure portal without a single line of code. In the second part, you will test, register, and deploy the model you trained in the first part using the Azure ML Python SDK.

In this lab we will be building a regression model to predict Taxi Fares in New York City. We will use a preprocessed labeled training data with features such as number of passengers, trip distance, datetime, holiday information and weather information.

Please complete Part 1 before Part 2.

## Part 1: Automated ML in Azure Portal

- Download the training data file [nyc-taxi-sample-data.csv](../starter-artifacts/python-notebooks/05-aml-automl/data) on your local disk
- Follow instructions at: [AutoML in Azure Portal](./automl-azure-portal/README.md)

## Part 2: Automated ML with Azure ML Python SDK

The second part of the lab can be performed using multiple environments, choose the environment in which you want to perform the lab below:

- [Azure Notebooks](./azure-notebooks/README.md)
- [Visual Studio Code](./visual-studio-code/README.md)
