# Technology overview

## Deep Learning with Azure Machine Learning service
Using the [main Python SDK](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) and the [Data Prep SDK](https://docs.microsoft.com/python/api/overview/azure/dataprep/intro?view=azure-dataprep-py) for Azure Machine Learning as well as open-source Python packages, you can build and train highly accurate machine learning and deep-learning models yourself in an Azure Machine Learning service Workspace. You can choose from many machine learning components available in open-source Python packages, such as the following examples:

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)
- [MXNet](https://mxnet.incubator.apache.org/)

After you have a model, you use it to create a container, such as Docker, that can be deployed locally for testing. After testing is done, you can deploy the model as a production web service in either Azure Container Instances or Azure Kubernetes Service. For more information, see the article on [how to deploy and where](https://docs.microsoft.com/azure/machine-learning/service/how-to-deploy-and-where).

Then you can manage your deployed models by using the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) or the [Azure portal](https://portal.azure.com). You can evaluate model metrics, retrain, and redeploy new versions of the model, all while tracking the model's experiments.

For deep neural network (DNN) training using TensorFlow, Azure Machine Learning provides a custom TensorFlow class of the Estimator. The Azure SDK's [TensorFlow estimator](https://docs.microsoft.com/python/api/azureml-train-core/azureml.train.dnn.tensorflow?view=azure-ml-py) (not to be conflated with the tf.estimator.Estimator class) enables you to easily submit TensorFlow training jobs for both single-node and distributed runs on Azure compute.

The TensorFlow Estimator also enables you to train your models at scale across CPU and GPU clusters of Azure VMs. You can easily run distributed TensorFlow training with a few API calls, while Azure Machine Learning will manage behind the scenes all the infrastructure and orchestration needed to carry out these workloads.

# Lab Overview
In this lab you will learn how to leverage Deep Learning technologies to scan through their vehicle specification documents to find compliance issues with new regulations.

Each document in the supplied training data set is a short text description of the component as documented by an authorized technician.

The contents include:
- Manufacture year of the component (e.g. 1985, 2010)
- Condition of the component (poor, fair, good, new)
- Materials used in the component (plastic, carbon fiber, steel, iron)

The compliance regulations dictate:
*Any component manufactured before 1995 or in fair or poor condition or made with plastic or iron is out of compliance.*

For example:
* Manufactured in 1985 made of steel in fair condition -> **Non-compliant**
* Good condition carbon fiber component manufactured in 2010 -> **Compliant**
* Steel component manufactured in 1995 in fair condition -> **Non-Compliant**

The labels present in this data are 0 for compliant, 1 for non-compliant.

The challenge with classifying text data is that deep learning models only undertand vectors (e.g., arrays of numbers) and not text. To encode the car component descriptions as vectors, we use an algorithm from Stanford called [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/). GloVe provides us pre-trained vectors that we can use to convert a string of text into a vector. 

## Next Steps

If you have not cloned this repository onto your local computer, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio-code`.

### Open the starting Python file
1. On your local computer expand the folder `03-aml-deep-learning`.
2. To run a lab, open Visual Studio Code by double clicking the starting python file: `deep-learning-with-AML.py`.
3. Confirm that your have setup `azure_automl` as your interpreter.
4. `deep-learning-with-AML.py` is the Python file you will step thru executing in this lab.
5. To execute each step click on `Run Cell` just above the block of code. 

### Follow the instructions within the python file to complete the lab
