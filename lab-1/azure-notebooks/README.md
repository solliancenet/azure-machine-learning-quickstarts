# Technology overview

## What are machine learning pipelines?
Using [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py), data scientists, data engineers, and IT professionals can collaborate on the steps involved in:
* Data preparation, such as normalizations and transformations
* Model training
* Model evaluation
* Deployment

The following diagram shows an example pipeline:

![piplines](./images/01.png)

# Lab Overview
The goal of this lab is to build a pipeline that demonstrate the basic data science workflow of data preparation, model training, and predictions. Azure Machine Learning, allows you to define distinct steps and make it possible to rerun only the steps you need as you tweak and test your workflow.

In this lab, we will develop a Car Battery Life prediction model using pandas and scikit-learn libraries. The work will be divided into 3 main pipeline steps: (1) preprocess the features, (2) train the model, and (3) make predictions. The preprocess step code will be also reused in the pipeline to make predictions on the test data. Thus you can centralize all your logic for preprocessing data for both train and test.

Using the pipeline steps, we will build and run two pipelines: (1) preprocess training data and train the model, and (2) preprocess the test data and make predictions using the trained model. Each of these pipelines is going to have implicit data dependencies and the example will demonstrate how AML make it possible to reuse previously completed steps and run only the modified or new steps in your pipeline.

The pipelines will be run on the Azure Machine Learning compute.

## Next Steps

If you have not cloned this repository to your Azure notebooks under the project `Aml-quickstarts`, do so now. All of the artifacts for this lab are located under `starter-artifacts/python-notebooks`.

### Open the starting Python Notebook
1. Within Azure Notebook, under `My Projects` open the project `Aml-quickstarts`. 
2. In the project expand the folder `01-aml-pipelines`.
5. To run a lab, you can start your project to run on “Free Compute”.
6. Open `pipelines-AML.ipynb`. This is the Python notebook you will step thru executing in this lab.
7. Confirm that Python 3.6 is set as your kernel for your notebook.

### Follow the instructions within the notebook to complete the lab
