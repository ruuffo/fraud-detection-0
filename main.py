import os
import zipfile

import kaggle
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def download_data():
    if not os.path.exists("data"):
        kaggle.api.dataset_download_files("mlg-ulb/creditcardfraud")

        with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
            zip_ref.extractall("data")

        zip_fraud_path = os.path.abspath("./creditcardfraud.zip")
        os.remove(zip_fraud_path)


download_data()

df = pd.read_csv("data/creditcard.csv")
TARGET_COLUMN_NAME = "Class"


def get_df_info(p_df: pd.DataFrame):
    return p_df.info(), p_df[TARGET_COLUMN_NAME].value_counts()


get_df_info(p_df=df)


def undersample(p_df: pd.DataFrame):
    df_non_fraud = p_df[p_df[TARGET_COLUMN_NAME] == 0]
    df_fraud = p_df[p_df[TARGET_COLUMN_NAME] == 1]

    df_non_fraud_undersampled = df_non_fraud.sample(df_fraud.shape[0], random_state=42)

    df_balanced_undersampled = pd.concat([df_fraud, df_non_fraud_undersampled])
    df_balanced_undersampled[TARGET_COLUMN_NAME].value_counts()
    return df_balanced_undersampled


df_balanced_undersampled = undersample(df)
df_balanced_undersampled[TARGET_COLUMN_NAME].value_counts()


def split_X_y(df_param: pd.DataFrame):
    X = df_param.drop(columns=[TARGET_COLUMN_NAME])
    y = df_param[TARGET_COLUMN_NAME]
    return X, y


def train_model_and_predict(df_param: pd.DataFrame) -> pd.Series:
    data, target = split_X_y(df_param)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    dec_tree = DecisionTreeClassifier()

    dec_tree.fit(X_train, y_train)

    y_predict = dec_tree.predict(X_test)
    return y_test, y_predict, dec_tree


y_test, y_predict, dec_tree = train_model_and_predict(df_balanced_undersampled)

pd.DataFrame(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))


def oversample(p_df: pd.DataFrame):
    X = p_df.drop(columns=[TARGET_COLUMN_NAME])
    y = p_df[TARGET_COLUMN_NAME]
    smote = SMOTE()
    df_data_oversampled, df_target_oversampled = smote.fit_resample(X, y)
    return df_data_oversampled, df_target_oversampled


df_data_oversampled, df_target_oversampled = oversample(df)
df_target_oversampled.value_counts()

df_oversampled = df_data_oversampled.assign(
    **{TARGET_COLUMN_NAME: df_target_oversampled}
)
y_test, y_predict, dec_tree = train_model_and_predict(df_oversampled)

pd.DataFrame(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
X, y = split_X_y(df)
y_predict = dec_tree.predict(X)

pd.DataFrame(confusion_matrix(y, y_predict))
print(classification_report(y, y_predict))
