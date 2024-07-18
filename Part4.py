
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# Title of the app
st.title('Data Analysis and Visualization App')

# Step 1: Load Data
st.header('Load Data')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # User inputs for header, separator, and encoding
        use_header = st.checkbox('Does the file have a header?', value=True)
        separator = st.selectbox('Select the separator', options=[',', ';', '	', '|'])
        encoding = st.selectbox('Select the encoding', options=['utf-8', 'latin1', 'iso-8859-1'])

        # Load the data
        if use_header:
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
        else:
            df = pd.read_csv(uploaded_file, sep=separator, header=None, encoding=encoding)

        # Display the first few rows of the dataframe to check if it loaded correctly
        st.subheader('First few rows of the data')
        st.write(df.head())
        
        # If no data is loaded, raise an error
        if df.empty:
            st.error("The loaded CSV file is empty or not properly formatted. Please check the file and try again.")
        else:
            # Step 2: Data Description
            st.header('Data Description')

            # Display first and last few rows
            st.subheader('First few rows of the data')
            st.write(df.head())

            st.subheader('Last few rows of the data')
            st.write(df.tail())

            # Step 3: Statistical Summary
            st.header('Statistical Summary')

            # Number of rows and columns
            st.subheader('Number of rows and columns')
            st.write(f'Number of rows: {df.shape[0]}')
            st.write(f'Number of columns: {df.shape[1]}')

            # Column names
            st.subheader('Column names')
            st.write(df.columns.tolist())

            # Number of missing values per column
            st.subheader('Missing values per column')
            st.write(df.isnull().sum())

            # Basic statistics
            st.subheader('Basic statistical summary')
            st.write(df.describe())

            # Step 4: Handle Missing Values
            st.header('Handle Missing Values')

            missing_value_method = st.selectbox(
                'Choose a method to handle missing values',
                ['Delete rows', 'Delete columns', 'Replace with mean', 'Replace with median', 'Replace with mode', 'KNN imputation']
            )

            if missing_value_method == 'Delete rows':
                df = df.dropna()
            elif missing_value_method == 'Delete columns':
                df = df.dropna(axis=1)
            elif missing_value_method == 'Replace with mean':
                imputer = SimpleImputer(strategy='mean')
                df[:] = imputer.fit_transform(df)
            elif missing_value_method == 'Replace with median':
                imputer = SimpleImputer(strategy='median')
                df[:] = imputer.fit_transform(df)
            elif missing_value_method == 'Replace with mode':
                imputer = SimpleImputer(strategy='most_frequent')
                df[:] = imputer.fit_transform(df)
            elif missing_value_method == 'KNN imputation':
                imputer = KNNImputer(n_neighbors=3)
                df[:] = imputer.fit_transform(df)

            st.subheader('Data after handling missing values')
            st.write(df)

            # Step 5: Normalize Data
            st.header('Normalize Data')

            normalization_method = st.selectbox(
                'Choose a normalization method',
                ['None', 'Min-Max normalization', 'Z-score standardization']
            )

            if normalization_method == 'Min-Max normalization':
                scaler = MinMaxScaler()
                df[:] = scaler.fit_transform(df)
            elif normalization_method == 'Z-score standardization':
                scaler = StandardScaler()
                df[:] = scaler.fit_transform(df)

            st.subheader('Data after normalization')
            st.write(df)

            # Step 6: Data Visualization
            st.header('Data Visualization')

            # Select column for visualization
            column_to_visualize = st.selectbox('Select a column to visualize', df.columns.tolist())

            # Histograms
            st.subheader('Histogram')
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column_to_visualize], kde=True)
            plt.xlabel(column_to_visualize)
            plt.title(f'Histogram of {column_to_visualize}')
            st.pyplot(plt)

            # Box plots
            st.subheader('Box Plot')
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[column_to_visualize])
            plt.ylabel(column_to_visualize)
            plt.title(f'Box Plot of {column_to_visualize}')
            st.pyplot(plt)

            # Step 7: Clustering
            st.header('Clustering')

            clustering_algorithm = st.selectbox(
                'Choose a clustering algorithm',
                ['K-Means', 'DBSCAN']
            )

            if clustering_algorithm == 'K-Means':
                num_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=num_clusters)
                df['Cluster'] = kmeans.fit_predict(df)
            elif clustering_algorithm == 'DBSCAN':
                eps = st.slider('Epsilon', min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.slider('Minimum samples', min_value=1, max_value=10, value=5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                df['Cluster'] = dbscan.fit_predict(df)

            st.subheader('Clustered Data')
            st.write(df)

            # Step 8: Prediction
            st.header('Prediction')

            prediction_task = st.selectbox(
                'Choose a prediction task',
                ['Regression', 'Classification']
            )

            target_column = st.selectbox('Select the target column', df.columns.tolist())
            feature_columns = st.multiselect('Select feature columns', [col for col in df.columns if col != target_column])

            X = df[feature_columns]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if prediction_task == 'Regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
                st.subheader('Regression Results')
                st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
            
                # Visualization
                st.subheader('Actual vs Predicted')
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted Values')
                st.pyplot(fig)
            elif prediction_task == 'Classification':
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader('Classification Results')
                st.write('Confusion Matrix')
                st.write(confusion_matrix(y_test, y_pred))
                st.write('Classification Report')
                st.write(classification_report(y_test, y_pred))
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info('Please upload a CSV file to get started.')

# To run the Streamlit app, save this script as `app.py` and use the following command:
# !streamlit run app.py
