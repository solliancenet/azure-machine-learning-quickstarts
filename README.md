# Azure Machine Learning Service Quickstarts

[Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml) is a cloud service for machine learning that support training, deploying, managing, and operationalizing (MLOps) models in Azure using Python SDK and CLI. Azure Machine Learning service also provides a visual interface (preview) to quickly, prepare data, and train machine learning models.

The Quickstarts are grouped into two parts: (1) Quickstarts for [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py), and (2) Quickstart for [Azure Machine Learning Visual Interface](https://docs.microsoft.com/en-us/azure/machine-learning/service/ui-quickstart-run-experiment).

## Azure Machine Learning Python SDK

The following set of quickstarts demonstrate a key set of Azure Machine Learning Services and provides instructions for you to preform them using a Python environment of your choice - Azure Notebooks or Visual Studio Code.

The following labs are available:
- [Lab 0:](./lab-0/README.md) Setting up your environment. If a lab environment has not be provided for you, this lab provides the instructions to get started in your own Azure Subscription.
- [Lab 1:](./lab-1/README.md) The goal of this lab is to build machine learning pipelines using Azure Machine Learning Python SDK that demonstrate the basic data science workflow of data preparation, model training, and predictions.
- [Lab 2:](./lab-2/README.md) The goal of this lab is to show Model interpretability with Azure Machine Learning service. You will learn how to explain why your model made the prediction it made by using the Azure Machine Learning Interpretability SDK. 
- [Lab 3:](./lab-3/README.md) In this lab you will train a deep learning model to classify the descriptions of car components as compliant or non-compliant. You will train the model on Azure Machine Learning Compute Cluster, download the trained model to your local computer, and make predictions.
- [Lab 4:](./lab-4/README.md) In this lab your will convert a Deep Learning model you trained in [Lab 3](./lab-3/README.md) to ONNX format and deploy the ONNX model as a web service to make inferences. You will also measure the speed of the ONNX runtime for making inferences and compare the speed of ONNX with Keras for making inferences.
- [Lab 5:](./lab-5/README.md) This lab is divided into two parts. In the first part, you learn how to create, run, and explore automated machine learning experiments in the Azure portal without a single line of code. In the second part, you will evaluate, register, and deploy the model you trained in the first part using the Azure ML Python SDK.

## Azure Machine Learning Visual Interface

Azure Machine Learning Visual Interface gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning Visual Interface also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications. To use Azure Machine Learning Studio, you do not need programming experience and following Quickstart will walk you through an exercise that will show how to process training data, create a model, train, score, and evaluate the model and finally deploy the trained model as a web service.
