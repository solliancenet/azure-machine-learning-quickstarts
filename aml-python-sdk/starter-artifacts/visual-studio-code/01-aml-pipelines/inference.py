import argparse
import os
import shutil
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor

print("In inference.py")

parser = argparse.ArgumentParser("inference")
parser.add_argument("--input", dest="input", required=True, type=str, help="input test features data")
parser.add_argument("--model", dest="model", required=True, type=str, help="model directory")
parser.add_argument("--output", dest="output", required=True, type=str, help="output directory")

args = parser.parse_args()

print("Argument 1: %s" % args.input)
print("Argument 2: %s" % args.model)
print("Argument 3: %s" % args.output)

# Load the test data
test_data = pd.read_csv(os.path.join(args.input, "test_features.csv"))
print('test data loaded!')

#load the model file
loaded_model = joblib.load(os.path.join(args.model, "gbr_model.joblib"))
print('model loaded!')

results = loaded_model.predict(test_data)
print('prediction done!')

if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    # Save the results
    with open(os.path.join(args.output, 'results.txt'), 'w') as f:
        for item in results:
            f.write("%s\n" % item)
    print('results saved!!')