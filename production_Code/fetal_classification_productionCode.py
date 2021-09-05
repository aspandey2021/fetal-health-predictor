"""
Author : Ashish Pandey

This script trains a classification model to read in a csv file which contains CTG results for fetal health,
preprocess, train over 70% of the data, test on the rest, output the
metrics and save the model in a pickle file.
"""

from datetime import datetime
import pickle, warnings
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support as score, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Preprocessing():

    def preProcess(self, df):
        # Dropping columns with name starting with 'histogram_' because they do not appear to be as the ...
        # ... features of a CTG exam and hence for simplification purposes, won't be considered for the analysis here.
        for col in df.columns:
            if 'histogram' in col:
                df.drop(columns=col, inplace=True)

        # Renaming some columns for ease of comprehension
        df.rename(columns={'abnormal_short_term_variability': '%_time_with_abnormal_short_term_variability',
                           'baseline value': 'FHR_baseline_value'}, inplace=True)
        df.rename(columns={
            'percentage_of_time_with_abnormal_long_term_variability': '%_time_with_abnormal_long_term_variability'},
            inplace=True)
        return df


class Classification(Preprocessing):

    def __init__(self):
        self.name = "Random Forest Classifier"

    # This function splits the data into train and test sets
    def splitData(self, df):
        # storing all the columns except the target into a variable
        featureCols = df.columns[:-1]

        # Splitting the data
        splitxy = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

        ## next() converts the generator object to an array
        train_idx, test_idx = next(splitxy.split(df[featureCols], df.fetal_health))

        # Create the dataframes
        x_train = df.loc[train_idx, featureCols]
        y_train = df.loc[train_idx, 'fetal_health']
        x_test = df.loc[test_idx, featureCols]
        y_test = df.loc[test_idx, 'fetal_health']

        return x_train, x_test, y_train, y_test

    # This function calculates and displays the error metrics
    def calculateScores(self, y_pred, y_prob, y_test, targetNames):
        print("\n-----------------\n")
        print("Calculating Scores ....\n")

        print(classification_report(y_test, y_pred, target_names=targetNames))
        precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')
        roc_auc_ = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr')
        accuracy_ = accuracy_score(y_test, y_pred)
        print("\n-----------------\n")
        print("The error metrics are as follows : \n")
        print('recall :', round(recall, 2))
        print('precision :', round(precision, 2))
        print('fscore :', round(fscore, 2))
        print('roc_auc_score :', round(roc_auc_, 2))
        print('accuracy :', round(accuracy_, 2))


    # This function trains and tests the model
    def modelClassifier(self, x_train, x_test, y_train, y_test):
        targetNames = ['class 1', 'class 2', 'class 3']

        # Defining the Model parameters
        model = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       n_estimators=300, n_jobs=-1, random_state=42,
                       warm_start=True)

        # Training the model and predicting
        print("\n-----------------\n")
        print("Model Name : ",self.name, "\n")
        print("Training the Model ...\n")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)

        # Calculating Error Metrics using multi-class support function
        self.calculateScores(y_pred, y_prob, y_test, targetNames)

        # Saving the file
        print("\nSaving File to disk...")
        with open("../fetal_classification_trained_model.pickle", 'wb') as fp:
            pickle.dump(model, fp)

        print("\n...Process Finished")


if __name__ == "__main__":
    start_time = datetime.now()
    print("\nReading in the data file...")
    df = pd.read_csv("../fetal_health.csv")

    obj1 = Classification()
    df_new = obj1.preProcess(df)
    x_train, x_test, y_train, y_test = obj1.splitData(df_new)
    obj1.modelClassifier(x_train, x_test, y_train, y_test)

    print("\n___ The Script took {} seconds ___".format(datetime.now() - start_time))
