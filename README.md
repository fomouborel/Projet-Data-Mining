# Projet-Data-Mining

### Overview
The goal of this project is to develop an interactive web application using Streamlit to analyze, clean, and visualize data. The application will also implement clustering algorithms to group similar objects in the dataset.

### Features
The project contains the following parts:

1. **Initial Data Exploration**
   - Data loading: Users can load their own dataset in CSV format. The header and the type of separation can be specified by the user.
   - Data description: Displays a preview of the first and last lines of data to check their correct loading.
   - Statistical summary: Provides a basic statistical summary of the data, including the number of lines and columns, the names of the columns, the number of missing values per column, etc.

2. **Data Pre-processing and Cleaning**
   - Managing missing values:
     - Deleting rows or columns with missing values.
     - Replacing missing values with the mean, median, or mode of the column.
     - Using imputation algorithms such as KNN imputation or simple imputation.
     - Users can choose the method they want to use to handle missing values.
   - Data normalization:
     - Min-Max normalization, which resizes the data to values between 0 and 1.
     - Z-score standardization, which resizes the data to have a mean of 0 and a standard deviation of 1.
     - Other normalization methods.
     - Users can choose the normalization method they want to use.

3. **Visualization of the Cleaned Data**
   - Histograms: Options for visualizing the distribution of data for each feature in the form of histograms.
   - Box plots: Visualize the distribution and outliers of each feature.

4. **Clustering or Prediction**
   - Clustering: Implements at least two clustering algorithms (e.g., k-means and DBSCAN). Users can choose the algorithm and set its parameters.
   - Prediction: Implements at least two prediction algorithms (e.g., regression or classification). Users can choose the algorithm and set its parameters.

5. **Learning Evaluation**
   - Visualization of clusters: Creates 2D or 3D scatter plots to visualize the clusters. Data points are colored according to their cluster membership.
   - Cluster statistics: Calculates and displays basic statistics on clusters, such as the number of data points in each cluster, the center of each cluster (for K-Means), the density of each cluster (for DBSCAN), etc.
   - Similar evaluation for prediction tasks.

### Requirements
- Python 3.7 or higher
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NumPy



  
**Installation**
   Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-mining-project.git
   cd data-mining-project```

### Installation
  python -m venv env
  source env/bin/activate  # On Windows use `env\Scripts\activate`

### Install the required packages:
   pip install -r requirements.txt

### Usage
   1.Run the Streamlit application: streamlit run streamlit_app.py
   2.Load your dataset in CSV format using the provided file uploader
   3.Follow the on-screen instructions to explore, clean, and visualize your data.
   4.Choose and apply clustering or prediction algorithms as needed.

**Acknowledgements**

    Special thanks to Issam Falih for their guidance and support.
    Thanks to all the team members for their hard work and collaboration.

