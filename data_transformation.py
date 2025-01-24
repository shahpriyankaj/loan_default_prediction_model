''' The python file to perform data preprocessing and data transformation before training the model. '''
# improt libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
'''
right:
annual_inc
total_bc_limit
tot_hi_cred_lim
tot_cur_bal

left:
bc_util
'''
# Helper functions
def get_numerical_features():
    # Get numerical features
    return ['loan_amnt', 'int_rate', 'annual_inc', 'percent_bc_gt_75', 'dti', 'inq_last_6mths', 'mths_since_recent_inq', 'total_bc_limit', 'tot_cur_bal', 'tot_hi_cred_lim']

def get_le_categorical_features():
    # get categorical features only for label encoding
    return ['emp_length_yr']

def get_ohe_categorical_features():
    # get categoring features for one hot encoding
    return ['home_ownership', 'purpose', 'term']

def convert_into_category(data):
    # Converting categorical feature's datatype to 'category'
    for column in get_ohe_categorical_features() + get_le_categorical_features():
        data[column] = data[column].astype('category')
    return data

# Data Preprocessing steps 
def feature_engineering(data):
    # Remove '%' symbol from int_rate and extract year from emp_lenght column string
    data['int_rate'] = data['int_rate'].str.replace('%', '').astype(float)
    data.loc[data['emp_length'] == '< 1 year', 'emp_length'] = '0 year'
    data['emp_length_yr'] = data['emp_length'].str.extract(r'(\d+)')
    data.drop(columns=['emp_length'], inplace=True)
    return data

def rare_category_combiner(data):
    # combine Less frequent into one category for home_ownership and purpose features
    data.loc[data['home_ownership'].isin(['OTHER', 'OWN']), 'home_ownership'] = 'OTHER'
    data.loc[~data['purpose'].isin(['debt_consolidation', 'credit_card']), 'purpose'] = 'other'
    return data

def preprocessor(data):
    # Function to perform all preprocessing together for the input data
    data = data.copy()
    data = feature_engineering(data)
    data = rare_category_combiner(data)
    data = data[get_numerical_features() + get_ohe_categorical_features() + get_le_categorical_features()]
    data = convert_into_category(data)
    return data

# Data Transformation
def numerical_transformation(x_train, x_test, test_dataset):
    '''
    For numerical features:
    - Impute missing values using Simple Imputer with 'median' strategy
    - power tranformation to handle skewed data using 'yeo-johnson' method
    - scaling using Robust scalar since it handles outlier as well

    This will perform on all x_train, x_test and testing dataset. We will fit & transform transformation on training data and transforn on x_test and test dataset

    input:
    x_train: training dataset in pandas dataframe
    x_test: validation set in pandas dataframe
    text_dataset: testing dataset in pandas dataframe

    output:
    transformed x_train, x_test, test_dataset in pandas dataframe
    '''
    imp = SimpleImputer(strategy='median')
    power = PowerTransformer(method='yeo-johnson')
    scaler = RobustScaler()
    for column in get_numerical_features():
        x_train[column] = imp.fit_transform(x_train[[column]])
        x_test[column] = imp.transform(x_test[[column]])
        test_dataset[column] = imp.transform(test_dataset[[column]])

        x_train[column] = power.fit_transform(x_train[[column]])
        x_test[column] = power.transform(x_test[[column]])
        test_dataset[column] = power.transform(test_dataset[[column]])

        x_train[column] = scaler.fit_transform(x_train[[column]])
        x_test[column] = scaler.transform(x_test[[column]])
        test_dataset[column] = scaler.transform(test_dataset[[column]])
    return x_train, x_test, test_dataset


def categorical_le_transformation(x_train, x_test, test_dataset):
    '''
    For categorical features:
    - impute missing values using Simple Imputer with 'most_frequent' strategy
    - Perform label encoding for emp_length feature to store the order

    This will perform on all x_train, x_test and testing dataset. We will fit & transform transformation on training data and transforn on x_test and test dataset

    input:
    x_train: training dataset in pandas dataframe
    x_test: validation set in pandas dataframe
    text_dataset: testing dataset in pandas dataframe

    output:
    transformed x_train, x_test, test_dataset in pandas dataframe
    '''
    imp = SimpleImputer(strategy='most_frequent')
    label = LabelEncoder()
    for column in get_le_categorical_features():
        x_train[column] = imp.fit_transform(x_train[[column]]).ravel() 
        x_test[column] = imp.transform(x_test[[column]]).ravel() 
        test_dataset[column] = imp.transform(test_dataset[[column]]).ravel() 

        x_train[column] = label.fit_transform(x_train[[column]])
        x_test[column] = label.transform(x_test[[column]])
        test_dataset[column] = label.transform(test_dataset[[column]])
    return x_train, x_test, test_dataset


def categorical_ohe_transformation(x_train, x_test, test_dataset):
    '''
    For categorical features except category feature required label encoding.
    - impute missing values using Simple Imputer with 'most_frequent' strategy
    - Perform one hot encoding to convert categorical columns into numerical form

    This will perform on all x_train, x_test and testing dataset. We will fit & transform transformation on training data and transforn on x_test and test dataset

    input:
    x_train: training dataset in pandas dataframe
    x_test: validation set in pandas dataframe
    text_dataset: testing dataset in pandas dataframe

    output:
    transformed x_train, x_test, test_dataset in pandas dataframe
    '''
    ohe_categorical_features = get_ohe_categorical_features()
    imp = SimpleImputer(strategy='most_frequent')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for column in ohe_categorical_features:
        x_train[column] = imp.fit_transform(x_train[[column]]).ravel()
        x_test[column] = imp.transform(x_test[[column]]).ravel() 
        test_dataset[column] = imp.transform(test_dataset[[column]]).ravel() 

    x_train_encoded = ohe.fit_transform(x_train[ohe_categorical_features])
    x_test_encoded = ohe.transform(x_test[ohe_categorical_features])
    test_dataset_encoded = ohe.transform(test_dataset[ohe_categorical_features])

    # Convert the result back to a DataFrame
    encoded_columns = ohe.get_feature_names_out(ohe_categorical_features)
    x_train_encoded = pd.DataFrame(x_train_encoded, columns=encoded_columns, index=x_train.index)
    x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoded_columns, index=x_test.index)
    test_dataset_encoded = pd.DataFrame(test_dataset_encoded, columns=encoded_columns, index=test_dataset.index)

    # Merge the encoded columns back into the original DataFrame
    x_train = x_train.drop(columns=ohe_categorical_features).join(x_train_encoded)
    x_test = x_test.drop(columns=ohe_categorical_features).join(x_test_encoded)
    test_dataset = test_dataset.drop(columns=ohe_categorical_features).join(test_dataset_encoded)
    return x_train, x_test, test_dataset


def data_transformation(x_train, x_test, test_dataset):
    '''
    The function to call all data transformation steps for numerical and categorical features.

    input:
    x_train: training dataset in pandas dataframe
    x_test: validation set in pandas dataframe
    text_dataset: testing dataset in pandas dataframe

    output:
    transformed x_train, x_test, test_dataset in pandas dataframe
    '''
    x_train, x_test, test_dataset = numerical_transformation(x_train, x_test, test_dataset)
    x_train, x_test, test_dataset = categorical_le_transformation(x_train, x_test, test_dataset)
    x_train, x_test, test_dataset = categorical_ohe_transformation(x_train, x_test, test_dataset)
    return x_train, x_test, test_dataset

'''
# Following code was developed to created data pipeline. But since it was not working as expected, I went with above approach.
def init_pipeline():
    # Combine pipelines into a ColumnTransformer
    transformer_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, get_numerical_features()),
            ('ohe_cat', categorical_pipeline, get_ohe_categorical_features()),
            ('le_cat', categorical_label_encoder_pipeline, get_le_categorical_features())
        ])
    return transformer_pipeline


# Data Transformation steps
class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        #self.le = LabelEncoder()
        self.label_encoders = {}
        self.columns = get_le_categorical_features()

    def fit(self, data):
        for column in self.columns:
            le = LabelEncoder()
            self.label_encoders[column] = le.fit(data[column].astype(str))
        return self

    def transform(self, data):
        data = data.copy()
        for column in self.columns:
            if column in data:
                data[column] = self.label_encoders[column].transform(data[column].astype(str))
        return data


# Pipeline for numerica features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('box_cox', PowerTransformer(method='yeo-johnson')),
    ('scaler', RobustScaler())
])

# Pipeline for categorical features
categorical_label_encoder_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', LabelEncodingTransformer())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
'''




