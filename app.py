import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Titre de l'application
st.title('Analyseur de fichier CSV')

# Formulaire pour téléverser un fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Options pour sélectionner l'entête et le séparateur
header_option = st.selectbox("Ligne d'entête", [None, 0, 1, 2, 3], index=1)
separator_option = st.selectbox("Type de séparateur", [",", ";", "\t", "|"], index=0)

# Liste des encodages courants
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
encoding_option = st.selectbox("Choisissez un encodage", encodings)

# Fonction pour afficher un résumé statistique
def display_summary(data):
    # Bascule pour afficher/masquer les premières lignes
    if st.checkbox("Afficher les premières lignes"):
        st.write("### Aperçu des premières lignes")
        st.dataframe(data.head())

    # Bascule pour afficher/masquer les dernières lignes
    if st.checkbox("Afficher les dernières lignes"):
        st.write("### Aperçu des dernières lignes")
        st.dataframe(data.tail())

    # Bascule pour afficher/masquer le résumé statistique
    if st.checkbox("Afficher le résumé statistique"):
        st.write("### Résumé statistique de base")
        st.write("Nombre de lignes:", data.shape[0])
        st.write("Nombre de colonnes:", data.shape[1])
        st.write("Noms des colonnes:", list(data.columns))
        st.write("Nombre de valeurs manquantes par colonne:")
        st.write(data.isnull().sum())

# Fonction pour supprimer les colonnes et lignes avec plus de 80% de valeurs manquantes
def remove_high_missing(data, threshold=0.8):
    # Supprimer les colonnes avec plus de 80% de valeurs manquantes
    col_thresh = int(threshold * len(data))
    data = data.dropna(axis=1, thresh=col_thresh)

    # Supprimer les lignes avec plus de 80% de valeurs manquantes
    row_thresh = int(threshold * len(data.columns))
    data = data.dropna(axis=0, thresh=row_thresh)

    return data

# Fonction pour gérer les valeurs manquantes
def handle_missing_values(df, method, columns):
    if method == 'Delete columns':
        return df.drop(columns=columns)
    elif method == 'Replace with mean':
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
    return df

# Fonction pour normaliser les données
def normalize_data(df, method):
    if method == 'Min-Max normalization':
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif method == 'Z-score standardization':
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

# Fonction pour permettre à l'utilisateur de choisir la méthode de gestion des valeurs manquantes
def choose_missing_value_method(data):
    qualitative_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    quantitative_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.write("### Méthodes de gestion des valeurs manquantes")

    st.write("#### Variables qualitatives")
    qual_method = st.selectbox(
        'Choisissez une méthode pour les variables qualitatives',
        ['Aucune', 'Delete columns', 'Replace with mode']
    )
    if qual_method != 'Aucune':
        data = handle_missing_values(data, qual_method, qualitative_cols)

    st.write("#### Variables quantitatives")
    quant_method = st.selectbox(
        'Choisissez une méthode pour les variables quantitatives',
        ['Aucune', 'Delete columns', 'Replace with mean', 'Replace with median', 'KNN imputation']
    )
    if quant_method != 'Aucune':
        data = handle_missing_values(data, quant_method, quantitative_cols)

    return data

# Chargement et affichage du fichier CSV si un fichier est téléversé
if uploaded_file is not None:
    try:
        # Tentative de chargement du fichier CSV avec l'encodage sélectionné
        data = pd.read_csv(uploaded_file, header=header_option, sep=separator_option, encoding=encoding_option)

        # Supprimer les colonnes et lignes avec plus de 80% de valeurs manquantes
        data = remove_high_missing(data)
        st.subheader('Data after deleted columns or line with at leat 80% NaN')
        st.write(data)

        # Affichage du résumé statistique
        display_summary(data)

        # Gestion des valeurs manquantes
        st.header('Handle Missing Values')
        data = choose_missing_value_method(data)

        #st.subheader('Data after handling missing values')
        #st.write(data)

        # Normalisation des données
        st.header('Normalize Data')
        normalization_method = st.selectbox(
            'Choose a normalization method',
            ['None', 'Min-Max normalization', 'Z-score standardization']
        )
        if normalization_method != 'None':
            data = normalize_data(data, normalization_method)

        st.subheader('Data after normalization')
        st.write(data)

    except UnicodeDecodeError:
        st.error(f"Erreur de décodage avec l'encodage {encoding_option}. Veuillez sélectionner un autre encodage.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
else:
    st.info('Please upload a CSV file to get started.')
