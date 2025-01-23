from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def get_numerical_features():
    return ['loan_amnt', 'int_rate', 'annual_inc', 'percent_bc_gt_75', 'dti', 'inq_last_6mths', 'mths_since_recent_inq', 'total_bc_limit']

def get_le_categorical_features():
    return ['emp_length_yr']

def get_ohe_categorical_features():
    return ['home_ownership', 'purpose', 'term']

def convert_into_category(data):
    for column in get_ohe_categorical_features() + get_le_categorical_features():
        data[column] = data[column].astype('category')
    return data

# Data Preprocessing steps 
def feature_engineering(data):
    data['int_rate'] = data['int_rate'].str.replace('%', '').astype(float)
    data.loc[data['emp_length'] == '< 1 year', 'emp_length'] = '0 year'
    data['emp_length_yr'] = data['emp_length'].str.extract(r'(\d+)')
    data.drop(columns=['emp_length'], inplace=True)
    return data


def rare_category_combiner(data):
    data.loc[data['home_ownership'].isin(['OTHER', 'OWN']), 'home_ownership'] = 'OTHER'
    data.loc[~data['purpose'].isin(['debt_consolidation', 'credit_card']), 'purpose'] = 'other'
    return data

def preprocessor(data):
    data = data.copy()
    data = feature_engineering(data)
    data = rare_category_combiner(data)
    data = data[get_numerical_features() + get_ohe_categorical_features() + get_le_categorical_features()]
    data = convert_into_category(data)
    return data

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
        self.le = LabelEncoder()
        #self.columns = get_le_categorical_features()

    def fit(self, data, y=None):
        self.le.fit(data['emp_length_yr'])
        return self

    def transform(self, data):
        self.le.transform(data['emp_length_yr'])
        return data


class ModelFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self,):
        return self

    def transform(self, data):
        return data[get_numerical_features() + get_ohe_categorical_features() + get_le_categorical_features()]
    

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_label_encoder_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    #('label_encoder', LabelEncodingTransformer())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])





