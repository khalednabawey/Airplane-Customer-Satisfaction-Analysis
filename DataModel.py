# importing neccsary libraries
import pickle
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from ipywidgets import widgets, interact

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectPercentile, SelectKBest, SelectFpr, f_classif, RFE, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.metrics import plot_confusion_matrix


class AirModel:

    def __init__(self, train_path, test_path) -> None:
        self.train_path = train_path
        self.test_path = test_path

    def __wrangle(self, path):

        df = pd.read_csv(path).drop("Unnamed: 0", axis=1)

        df.fillna(0, inplace=True)
        df.drop("id", axis=1, inplace=True)

        satisfaction_columns = df.columns[6:-4]
        df['overall_rating'] = round(df[satisfaction_columns].mean(axis=1), 2)

        # Instantiate LabelEncoder
        enc = LabelEncoder()
        # Encoding the categorical features in the training and test data
        for column in df.select_dtypes("O").columns:
            df[f'{column}_enc'] = enc.fit_transform(df[column])
            df.drop(column, axis=1, inplace=True)

        # Check that all columns are numeric
        if df.select_dtypes("O").columns.any():
            print("Non-numeric columns found:", df.select_dtypes("O").columns)
            return None  # or handle this case as you find suitable

        # Inistantiating the scaler object
        scaler = StandardScaler()
        # Fitting the scaler to the data and transforming it.
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        return df

    def split_data(self):

        train_df = self.__wrangle(self.train_path)
        test_df = self.__wrangle(self.test_path)

        X_train = train_df.drop("satisfaction_enc", axis=1)
        y_train = pd.Series(train_df['satisfaction_enc']).astype('category')

        X_test = test_df.drop("satisfaction_enc", axis=1)
        y_test = pd.Series(test_df['satisfaction_enc']).astype('category')

        return X_train, y_train, X_test, y_test

    def save_model(self, directory):

        # Ensure the path to the directory exists, create if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Path to save the model
        file_path = os.path.join(directory, "rf_model.pth")

        # Open the file in binary write mode and save the model using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.rf_model, f)

        return file_path

    def fit_transform(self):

        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()

        self.selector = SelectPercentile(score_func=f_classif, percentile=60)

        X_train_selected = self.selector.fit_transform(
            self.X_train, self.y_train
        )

        # Inistansiate the rf classifier object
        self.rf_model = RandomForestClassifier()

        # Get the names of the selected columns
        selected_columns = self.X_train.columns[self.selector.get_support()]

        # Show the selected columns
        print("Selected Columns:", list(selected_columns))

        # Fitting the model to the selected training data
        self.rf_model.fit(X_train_selected, self.y_train)

        # Saving the model for further use
        self.model_path = self.save_model(
            r"E:\edu\Data Science Tools\Project DataSets\Airline Passenger Satisfaction"
        )

    def predict(self, X_test=None):

        if not X_test:
            X_test = self.X_test

        X_test_selected = self.selector.transform(X_test)

        y_pred = self.rf_model.predict(X_test_selected)

        return y_pred

    def show_res(self):

        y_pred = self.predict()

        # Accuracy
        accuracy_train = accuracy_score(
            self.y_train, self.rf_model.predict(self.X_train_selected))
        accuracy_test = accuracy_score(self.y_test, y_pred)
        print("Accuracy:")
        print("\tTraining -> ", accuracy_train, end="\t")
        print("\tTesting -> ", accuracy_test.round(4))

        # Precision
        print("Precision:")
        print("\tTraining -> ", precision_score(self.y_train,
              self.rf_model.predict(self.X_train_selected)), end="\t")
        print("\tTesting -> ", precision_score(self.y_test, y_pred).round(4))

        # Recall
        print("Recall:")
        print("\tTraining -> ", recall_score(self.y_train,
              self.rf_model.predict(self.X_train_selected)), end="\t")
        print("\tTesting -> ", recall_score(self.y_test, y_pred).round(4))

    def run(self, show_result=False):

        self.split_data()

        self.fit_transform()
        self.predict()
        if show_result:
            self.show_res()


