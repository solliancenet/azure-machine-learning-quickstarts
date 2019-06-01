import argparse
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

print("In train.py")
print("As a data scientist, this is where I write my training code.")

parser = argparse.ArgumentParser("train")

parser.add_argument("--input", type=str, help="input directory", dest="input", required=True)
parser.add_argument("--output", type=str, help="output directory", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.input)
print("Argument 2: %s" % args.output)

# Load your processed features and outputs
X = pd.read_csv(os.path.join(args.input, 'features.csv'))
y = pd.read_csv(os.path.join(args.input, 'outputs.csv'))

# Train your model
X_train, X_eval, y_train, y_eval = train_test_split(X , y, random_state = 0)
tree = GradientBoostingRegressor(random_state = 0, max_features='auto', learning_rate=0.1, max_depth=6)
tree.fit(X_train, y_train)

# Output the train and evaluation accuracies
print('Accuracy of GBR on training set: {:.2f}'.format(tree.score(X_train, y_train)))
print('Accuracy of GBR on evaluation set: {:.2f}'.format(tree.score(X_eval, y_eval)))

# Save the model
if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    output_filename = os.path.join(args.output, 'gbr_model.joblib')
    joblib.dump(tree, output_filename) 
    print('Model file gbr_model.joblib saved!')

