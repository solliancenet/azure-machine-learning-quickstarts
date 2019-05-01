# Technology overview

## Model interpretability with Azure Machine Learning service
Machine learning interpretability is important in two phases of machine learning development cycle:

* During training: Model designers and evaluators require interpretability tools to explain the output of a model to stakeholders to build trust. They also need insights into the model so that they can debug the model and make decisions on whether the behavior matches their objectives. Finally, they need to ensure that the model is not biased.
* During inferencing: Predictions need to be explainable to the people who use your model. For example, why did the model deny a mortgage loan, or predict that an investment portfolio carries a higher risk?

The [Azure Machine Learning Interpretability Python SDK](https://docs.microsoft.com/en-us/python/api/azureml-explain-model/?view=azure-ml-py) incorporates technologies developed by Microsoft and proven third-party libraries (for example, SHAP and LIME). The SDK creates a common API across the integrated libraries and integrates Azure Machine Learning services. Using this SDK, you can explain machine learning models globally on all data, or locally on a specific data point using the state-of-art technologies in an easy-to-use and scalable fashion.

# Lab Overview
Your goal in this lab is to predict how much time a car battery has left until it is expected to fail and use the automated machine learning model explainer to retrieve important features for predictions. You are provided training data that includes telemetry from different vehicles, as well as the expected battery life that remains. From this you will train a model that given just the vehicle telemetry predicts the expected battery life. You will use automlexplainer module to retrieve feature importance for all iterations (global explanation) and then explain the model with previously unseen test data (specific explanation).

## Next Steps

If you have not cloned this repository onto your local computer, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio-code`.

### Open the starting Python file
1. On your local computer expand the folder `02-aml-interpretability`.
2. To run a lab, open Visual Studio Code by double clicking the starting python file: `interpretability-with-AML.py`.
3. Confirm that your have setup `azure_automl` as your interpreter.
4. `interpretability-with-AML.py` is the Python file you will step thru executing in this lab.
5. To execute each step click on `Run Cell` just above the block of code. 

### Follow the instructions within the python file to complete the lab
