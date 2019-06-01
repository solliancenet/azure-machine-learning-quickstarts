import argparse
import os
import shutil
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib

print("In preprocess.py")

parser = argparse.ArgumentParser("preprocess")
parser.add_argument("--process_mode", type=str, help="process mode: train or inference")
parser.add_argument("--input", type=str, help="input raw data")
parser.add_argument("--preprocessors", type=str, help="output / input directory for preprocessors")
parser.add_argument("--output", type=str, help="output directory for processed data")

args = parser.parse_args()

print("Argument 1: %s" % args.process_mode)
print("Argument 2: %s" % args.input)
print("Argument 3: %s" % args.preprocessors)
print("Argument 4: %s" % args.output)

if(args.process_mode == 'train'):
    # Load the training data
    train_data = pd.read_csv(args.input)
    print('train data loaded!')

    # Preprocess the training data
    le_province = preprocessing.LabelEncoder()
    le_province.fit(train_data.Province.unique())
    train_data['Province_code'] = le_province.transform(train_data.Province)

    le_region = preprocessing.LabelEncoder()
    le_region.fit(train_data.Region.unique())
    train_data['Region_code'] = le_region.transform(train_data.Region)

    le_manufacture_month = preprocessing.LabelEncoder()
    le_manufacture_month.fit(train_data.Manufacture_Month.unique())
    train_data['Manufacture_Month_code'] = le_manufacture_month.transform(train_data.Manufacture_Month)

    le_manufacture_year = preprocessing.LabelEncoder()
    le_manufacture_year.fit(train_data.Manufacture_Year.unique())
    train_data['Manufacture_Year_code'] = le_manufacture_year.transform(train_data.Manufacture_Year)

    train_data['Car_Has_EcoStart_code'] = train_data.Car_Has_EcoStart.apply(lambda x: 1 if x else 0)

    X = train_data.drop(['Province','Region', 'Manufacture_Month', 
                                    'Manufacture_Year', 'Car_Has_EcoStart', 'Survival_In_Days'], axis=1)
    X.fillna(-50, inplace=True)
    y = train_data.Survival_In_Days
    
    print('preprocessing train data done!')

    if not (args.output is None):
        os.makedirs(args.output, exist_ok=True)
        # Save the features and output values
        X.to_csv(os.path.join(args.output, "features.csv"), header=True, index=False)
        y.to_csv(os.path.join(args.output, "outputs.csv"), header=True, index=False)
        print('X and y train files saved!')
        # Save the LabelEncoder's for inference pipeline
        os.makedirs(args.preprocessors, exist_ok=True)
        joblib.dump(le_province, os.path.join(args.preprocessors, "province.save"))
        joblib.dump(le_region, os.path.join(args.preprocessors, "region.save"))
        joblib.dump(le_manufacture_month, os.path.join(args.preprocessors, "manufacture_month.save"))
        joblib.dump(le_manufacture_year, os.path.join(args.preprocessors, "manufacture_year.save"))
        print('preprocessors saved!')

if(args.process_mode == 'inference'):
    # Load the test data
    test_data = pd.read_csv(args.input)
    print('test data loaded!')

    # Preprocess the test data
    scaler = joblib.load(os.path.join(args.preprocessors, 'province.save'))
    test_data['Province_code'] = scaler.transform(test_data.Province)

    scaler = joblib.load(os.path.join(args.preprocessors, 'region.save'))
    test_data['Region_code'] = scaler.transform(test_data.Region)

    scaler = joblib.load(os.path.join(args.preprocessors, 'manufacture_month.save'))
    test_data['Manufacture_Month_code'] = scaler.transform(test_data.Manufacture_Month)

    scaler = joblib.load(os.path.join(args.preprocessors, 'manufacture_year.save'))
    test_data['Manufacture_Year_code'] = scaler.transform(test_data.Manufacture_Year)

    test_data['Car_Has_EcoStart_code'] = test_data.Car_Has_EcoStart.apply(lambda x: 1 if x else 0)

    X = test_data.drop(['Car_ID', 'Battery_Age', 'Province','Region', 'Manufacture_Month', 
                        'Manufacture_Year', 'Car_Has_EcoStart'], axis=1)

    X.fillna(-50, inplace=True)

    print('preprocessing test data done!')
    
    if not (args.output is None):
        os.makedirs(args.output, exist_ok=True)
        # Save the features
        X.to_csv(os.path.join(args.output, "test_features.csv"), header=True, index=False)
        print('X test file saved!')
    