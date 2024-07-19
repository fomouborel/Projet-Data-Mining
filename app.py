from xml.etree.ElementInclude import include

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import math
from sklearn.linear_model import LinearRegression

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as pltnear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as pxleImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Title of the application
st.title('CSV File Analyzer')

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = ["Upload Data and Describe", "Process and Clean the Data", "Visualize Data", "Clustering and Prediction"]
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
    quantitative_cols = data.select_dtypes(include=(['int64', 'float64'])).columns.tolist()

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




# Function to plot histograms
def plot_histograms(data):
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(data[col], bins=30, kde=True, ax=axs[i])
        axs[i].set_title(f'Histogram of {col}')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    st.pyplot(fig)

# Function to plot box plots
def plot_box_plots(data):
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=data[col], ax=axs[i])
        axs[i].set_title(f'Box plot of {col}')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    st.pyplot(fig)


# Function to plot dimension reduction
def plot_dimension_reduction(data, target_column):
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 1:
        X = data[num_cols].dropna()
        y = data[target_column]

        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target column must be numeric for dimension reduction.")
            return


        pca = PCA(n_components=2)

        mds = MDS(n_components=2)
        lle = LocallyLinearEmbedding(n_components=2)

        X_pca = pca.fit_transform(X)

        X_mds = mds.fit_transform(X)
        X_lle = lle.fit_transform(X)

        st.session_state.X_pca = X_pca

        st.session_state.X_mds = X_mds
        st.session_state.X_lle = X_lle

        fig, axs = plt.subplots(2, 2, figsize=(20, 12))

        scatter = axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
        axs[0, 0].set_title('PCA')


        scatter = axs[1, 0].scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis', s=50)
        axs[1, 0].set_title('MDS')

        scatter = axs[1, 1].scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap='viridis', s=50)
        axs[1, 1].set_title('LLE')

        fig.colorbar(scatter, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1)

        st.pyplot(fig)
    else:
        st.error("Not enough numerical columns for dimension reduction.")
# Function to plot clusters
def plot_clusters(data, labels, algorithm, centers=None):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    plt.figure(figsize=(10, 6))
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    if centers is not None:
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(f'Clusters found by {algorithm}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    st.pyplot(plt)

# Function to plot the elbow method
def plot_elbow(data):
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sse, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    st.pyplot(plt)

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

elif step == "Visualize Data":
    if st.session_state.data is not None:
        data = st.session_state.data
        st.header('Visualize Data')

        st.subheader('Data Distribution')
        st.write(data.describe(include='all'))

        st.subheader('Histograms')
        plot_histograms(data)

        st.subheader('Box Plots')
        plot_box_plots(data)


        st.subheader('Dimension Reduction')
        target_column = st.selectbox('Choose the target column for dimension reduction', data.select_dtypes(include=['float64', 'int64']).columns)
        plot_dimension_reduction(data, target_column)


    else:
        st.info("Please complete the previous steps first.")


elif step == "Clustering and Prediction":
    if st.session_state.data is not None:
        data = st.session_state.data

        # Select only numeric columns for clustering
        numeric_data = data.select_dtypes(include=['int64', 'float64'])

        st.header('Clustering')
        clustering_algorithm = st.selectbox(
            'Choose a clustering algorithm',
            ['K-means', 'DB-SCAN']
        )
        if clustering_algorithm == 'K-means':
            st.subheader('Elbow Method for Optimal k')
            plot_elbow(numeric_data)
            n_clusters = st.number_input('Number of clusters', min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(numeric_data)
            
            plot_clusters(numeric_data, labels, 'K-means', kmeans.cluster_centers_)
        elif clustering_algorithm == 'DB-SCAN':
            eps = st.number_input('Epsilon (eps)', min_value=0.1, max_value=10.0, value=0.5)
            min_samples = st.number_input('Minimum samples', min_value=1, max_value=20, value=5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(numeric_data)
            plot_clusters(numeric_data, labels, 'DB-SCAN')

        st.header('Prediction')
        target_column = st.selectbox('Choose the target column', data.columns)
        st.write(f'Target column selected: {target_column}')

        if st.button('Run Linear Regression'):
            # Prepare the data for regression
            X = data.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64'])
            y = data[target_column]

            # Handle any remaining missing values in X and y
            X = SimpleImputer(strategy='mean').fit_transform(X)
            y = SimpleImputer(strategy='mean').fit_transform(y.values.reshape(-1, 1)).ravel()

            # Run linear regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Display regression results
            st.write('### Regression Results')
            st.write(f'R^2 Score: {model.score(X, y):.2f}')
            st.write(f'Coefficients: {model.coef_}')
            st.write(f'Intercept: {model.intercept_}')

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.3)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            st.pyplot(plt)

        if st.button('Logistic Regression'):
             # Prepare the data

            X = data.drop(columns=[target_column]).select_dtypes(include=['number']).dropna()
            y = data[target_column].dropna()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model= LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred =model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            st.write(f"### Accuracy: {accuracy:.2f}")
            st.write("### Confusion Matrix")
            st.write(conf_matrix)

            st.write("### Classification Report")
            st.write(pd.DataFrame(class_report).transpose())

            # Visualize the confusion matrix
            fig = px.imshow(conf_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig)

        if st.button('Random Forest'):
             # Prepare the data

            X = data.drop(columns=[target_column]).select_dtypes(include=['number']).dropna()
            y = data[target_column].dropna()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model= RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred =model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            st.write(f"### Accuracy: {accuracy:.2f}")
            st.write("### Confusion Matrix")
            st.write(conf_matrix)

            st.write("### Classification Report")
            st.write(pd.DataFrame(class_report).transpose())

            # Visualize the confusion matrix
            fig = px.imshow(conf_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig)
        if st.button('k-Nearest Neighbors (k-NN)'):
             # Prepare the data

            X = data.drop(columns=[target_column]).select_dtypes(include=['number']).dropna()
            y = data[target_column].dropna()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model= KNeighborsClassifier()
            model.fit(X_train, y_train)
            y_pred =model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            st.write(f"### Accuracy: {accuracy:.2f}")
            st.write("### Confusion Matrix")
            st.write(conf_matrix)

            st.write("### Classification Report")
            st.write(pd.DataFrame(class_report).transpose())

            # Visualize the confusion matrix
            fig = px.imshow(conf_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig)

    else:
        st.info("Please upload and process a CSV file first.")



