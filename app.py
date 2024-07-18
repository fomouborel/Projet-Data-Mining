import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# Title of the application
st.title('CSV File Analyzer')

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = ["Upload Data and Describe", "Process and Clean the Data"]
step = st.sidebar.radio("Go to", steps)

# Function to display a statistical summary
def display_summary(data):
    if st.checkbox("Show first rows"):
        st.write("### Preview of the first rows")
        st.dataframe(data.head())

    if st.checkbox("Show last rows"):
        st.write("### Preview of the last rows")
        st.dataframe(data.tail())

    if st.checkbox("Show statistical summary"):
        st.write("### Basic Statistical Summary")
        st.write("Number of rows:", data.shape[0])
        st.write("Number of columns:", data.shape[1])
        st.write("Column names:", list(data.columns))
        st.write("Number of missing values per column:")
        st.write(data.isnull().sum())

# Function to remove rows and columns with high missing values and duplicates
def remove_high_missing_and_duplicates(data, col_threshold=0.8, row_threshold=0.85):
    # Remove columns with at least 80% missing values
    col_thresh = int((1 - col_threshold) * len(data))
    data = data.dropna(axis=1, thresh=col_thresh)

    # Remove rows with at least 80% missing values
    row_thresh = int((1 - row_threshold) * len(data.columns))
    data = data.dropna(axis=0, thresh=row_thresh)
    
    # Remove duplicate rows
    data = data.drop_duplicates()

    return data

# Function to handle missing values
def handle_missing_values(df, method, columns):
    if method == 'Replace with mean':
        imputer = SimpleImputer(strategy='mean')
        df[columns] = imputer.fit_transform(df[columns])
    elif method == 'Replace with median':
        imputer = SimpleImputer(strategy='median')
        df[columns] = imputer.fit_transform(df[columns])
    elif method == 'Replace with mode':
        imputer = SimpleImputer(strategy='most_frequent')
        df[columns] = imputer.fit_transform(df[columns])
    elif method == 'KNN imputation':
        imputer = KNNImputer(n_neighbors=3)
        df[columns] = imputer.fit_transform(df[columns])
    elif method == 'Regression imputation':
        for column in columns:
            if df[column].isnull().sum() > 0:
                df, score = regression_imputation(df, column)
                st.write(f"Regression score for column {column}: {score:.2f}")
    return df

# Function for regression imputation
def regression_imputation(df, target_column):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df = numeric_df.dropna(axis=1, how='all')
    train = numeric_df[numeric_df[target_column].notnull()]
    test = numeric_df[numeric_df[target_column].isnull()]

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    
    df.loc[df[target_column].isnull(), target_column] = model.predict(X_test)

    return df, score

# Function to normalize data
def normalize_data(df, method):
    if method == "Min-Max normalization":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif method == "Z-score standardization":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# Function to allow users to choose the method for handling missing values
def choose_missing_value_method(data):
    qualitative_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    quantitative_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.write("### Methods for Handling Missing Values")

    st.write("#### Qualitative Variables")
    qual_method = st.selectbox(
        'Choose a method for qualitative variables',
        ['None', 'Replace with mode']
    )
    if qual_method != 'None':
        data = handle_missing_values(data, qual_method, qualitative_cols)

    st.write("#### Quantitative Variables")
    quant_method = st.selectbox(
        'Choose a method for quantitative variables',
        ['None', 'Replace with mean', 'Replace with median', 'KNN imputation', 'Regression imputation']
    )
    if quant_method != 'None':
        data = handle_missing_values(data, quant_method, quantitative_cols)

    return data

# Loading and displaying the CSV file if uploaded
if 'data' not in st.session_state:
    st.session_state.data = None

if step == "Upload Data and Describe":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Options to select the header and separator
    header_option = st.selectbox("Header row", [None, 0, 1, 2, 3], index=1)
    separator_option = st.selectbox("Separator type", [",", ";", "\t", "|"], index=0)

    # List of common encodings
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    encoding_option = st.selectbox("Choose an encoding", encodings)

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, header=header_option, sep=separator_option, encoding=encoding_option)
            data = remove_high_missing_and_duplicates(data)
            st.session_state.data = data
            st.subheader('Data after deleting columns or rows with at least 80% NaN and removing duplicates')
            st.write(data)
            display_summary(data)

        except UnicodeDecodeError:
            st.error(f"Decoding error with encoding {encoding_option}. Please select a different encoding.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif step == "Process and Clean the Data":
    if st.session_state.data is not None:
        data = st.session_state.data
        st.header('Handle Missing Values')
        data = choose_missing_value_method(data)
        st.session_state.data = data
        st.subheader('Data after handling missing values')
        st.write(data)

        st.header('Normalize Data')
        normalization_method = st.selectbox(
            'Choose a normalization method',
            ['None', 'Min-Max normalization', 'Z-score standardization']
        )
        if normalization_method != 'None':
            data = normalize_data(data, normalization_method)
            st.session_state.data = data

        st.subheader('Data after normalization')
        st.write(data)

        st.header('Summary of Data')
        display_summary(data)
    else:
        st.info("Please upload a CSV file first.")
