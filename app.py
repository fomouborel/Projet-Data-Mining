import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.metrics.pairwise import euclidean_distances

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
target_names = iris.target_names

st.write('### Iris Dataset')
st.write(pd.DataFrame(data=np.c_[X, Y], columns=iris.feature_names + ['target']))

# Perform PCA on the dataset
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Perform LDA on the dataset
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit_transform(X, Y)

st.title('PCA and LDA on Dataset')

# PCA Plot
st.write('## PCA of Iris Dataset')
fig, ax = plt.subplots()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
               label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('PCA of Iris dataset')
st.pyplot(fig)

# Splitage des données en 80/20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5, stratify=Y)

# Check the quality of the training sample using a contingency table
contingency_table = pd.crosstab(index=Y_train, columns='count')
st.write("## Contingency Table for Training Sample")
st.write(contingency_table)

# Create DataFrames for the training and test samples
st.write("## Attribute transformation")
train_data = pd.DataFrame(data=np.c_[X_train, Y_train], columns=iris.feature_names + ['Species'])
test_data = pd.DataFrame(data=np.c_[X_test, Y_test], columns=iris.feature_names + ['Species'])

don = train_data
don['Species'] = don['Species'].astype('str')
don.loc[don['Species'] == '2.0', 'Species'] = 'a'
don.loc[don['Species'] == '1.0', 'Species'] = 'e'
don.loc[don['Species'] == '0.0', 'Species'] = 's'
don['Species'] = don['Species'].astype('category')

st.write(don)

# Fit the LDA model with 2 components
lda1 = LinearDiscriminantAnalysis(n_components=2)
lda1.fit(X_train, Y_train)
y_pred = lda1.predict(X_test)

# Transform the training data
X_train_lda = lda1.transform(X_train)

# Plot LDA results
st.write("## Let's plot this supervised learning method")
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_train_lda[Y_train == i, 0], X_train_lda[Y_train == i, 1], alpha=.8, color=color,
               label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('LDA with train dataset')
st.pyplot(fig)

# Evaluation/statistiques pour evaluer notre model
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
st.write(f"Accuracy of LDA model: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)

# PART 2: Example with Euclidean distance
st.write("The euclidean_distances() function from sklearn.metrics.pairwise computes the Euclidean "
         "distance between pairs of points in two arrays. The Euclidean distance between two points "
         "p and q in n-dimensional space is defined as:")
# LaTeX string for the Euclidean distance formula
euclidean_formula = r"""
d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
"""
st.write(euclidean_formula)

st.write("Example")
X_example = [[0, 1], [1, 1]]
st.write("X = [[0, 1], [1, 1]], euclidean_distances(X, X), euclidean_distances(X, [[0, 0]])")
st.write(euclidean_distances(X_example, X_example))
st.write(euclidean_distances(X_example, [[0, 0]]))

# Apply MDS to the Iris dataset
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
fig, ax = plt.subplots()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_mds[Y == i, 0], X_mds[Y == i, 1], alpha=.8, color=color,
               label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('MDS of Iris dataset')
st.pyplot(fig)

# Apply Isomap to the Iris dataset
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)
fig, ax = plt.subplots()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_isomap[Y == i, 0], X_isomap[Y == i, 1], alpha=.8, color=color,
               label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('Isomap of Iris dataset')
st.pyplot(fig)

# Apply LLE to the Iris dataset
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)
fig, ax = plt.subplots()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_lle[Y == i, 0], X_lle[Y == i, 1], alpha=.8, color=color,
               label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('LLE of Iris dataset')
st.pyplot(fig)

## PART 3
from minisom import MiniSom

## PART 3

# Initialize and train the MiniSom
n_neurons = 9
m_neurons = 9
som = MiniSom(n_neurons, m_neurons, X.shape[1], sigma=1.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=0)
som.random_weights_init(X)
som.train(X, num_iteration=100, verbose=True)  # random training

# Quantization error
quantization_error = som.quantization_error(X)
st.write(f"Quantization error: {quantization_error}")

# Visualization setup
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# 1. Number of data points per neuron
neuron_counts = np.zeros((n_neurons, m_neurons))
for x in X:
    w = som.winner(x)
    neuron_counts[w] += 1

im1 = axs[0, 0].imshow(neuron_counts.T, cmap='Blues', origin='lower')
axs[0, 0].set_title('Number of Data Points per Neuron')
fig.colorbar(im1, ax=axs[0, 0])

# 2. Distance map (U-Matrix)
distance_map = som.distance_map()
im2 = axs[0, 1].imshow(distance_map.T, cmap='bone_r', origin='lower')
axs[0, 1].set_title('Distance Between Neighbors')
fig.colorbar(im2, ax=axs[0, 1])

# 3. Attribute description per neuron
weights = som.get_weights()
for i, attribute in enumerate(iris.feature_names):
    row = 1
    col = i % 2
    ax = axs[row, col]
    im = ax.imshow(weights[:, :, i].T, cmap='coolwarm', origin='lower')
    ax.set_title(f'Mean {attribute} per Neuron')
    fig.colorbar(im, ax=ax)

plt.tight_layout()
st.pyplot(fig)

# Load the data 'Waveform.csv' from teams and analyze them with the describe() and corr() functions from pandas
st.write("#### Load the data ’Waveform.csv’ from teams and analyze them with the describe() and corr() functions from pandas.")

# Assuming the file path is given, replace with the correct path to your CSV file
file_path = r"C:\Users\khassim\Cours_master1\S8\dataMining\pythonProject\waveform.csv"

# Load the dataset
waveform = pd.read_csv(file_path, header=None)

# Display dataset information
st.write("### Waveform Dataset Description")
st.write(waveform.describe())

# Correlation matrix
waveform_corr = waveform.corr()
st.write("### Waveform Dataset Correlation Matrix")
st.write(waveform_corr)


# Separate features and target
X_waveform = waveform.iloc[:, :-1]
y_waveform = waveform.iloc[:, -1]

# Apply PCA
pca = PCA(n_components=2)
X_waveform_pca = pca.fit_transform(X_waveform)

# Plot PCA results
st.write("### PCA of Waveform Data")
fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
scatter = ax_pca.scatter(X_waveform_pca[:, 0], X_waveform_pca[:, 1], c=y_waveform, cmap='viridis')
ax_pca.set_xlabel('Principal Component 1')
ax_pca.set_ylabel('Principal Component 2')
ax_pca.set_title('PCA of Waveform Data')
fig_pca.colorbar(scatter, ax=ax_pca)
st.pyplot(fig_pca)

# Commentaires sur les résultats de l'ACP
st.write("#### The PCA plot shows a good separation between the three clusters. This suggests that the features in the waveform dataset are effective in distinguishing between the different classes.")

# Initialize and train the SOM
n_neurons = 9
m_neurons = 9
som = MiniSom(n_neurons, m_neurons, X_waveform.shape[1], sigma=1.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=0)

som.random_weights_init(X_waveform.values)
som.train(X_waveform.values, num_iteration=100, verbose=True)

# Quantization error
quantization_error = som.quantization_error(X_waveform.values)
st.write(f"### Quantization error: {quantization_error}")

# Visualization setup
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# 1. Number of data points per neuron
neuron_counts = np.zeros((n_neurons, m_neurons))
for x in X_waveform.values:
    w = som.winner(x)
    neuron_counts[w] += 1

im1 = axs[0, 0].imshow(neuron_counts.T, cmap='Blues', origin='lower')
axs[0, 0].set_title('Number of Data Points per Neuron')
fig.colorbar(im1, ax=axs[0, 0])

# 2. Distance map (U-Matrix)
distance_map = som.distance_map()
im2 = axs[0, 1].imshow(distance_map.T, cmap='bone_r', origin='lower')
axs[0, 1].set_title('Distance Between Neighbors')
fig.colorbar(im2, ax=axs[0, 1])

# 3. Attribute description per neuron
weights = som.get_weights()
for i in range(X_waveform.shape[1]):
    ax = axs[1, i % 2]
    im = ax.imshow(weights[:, :, i].T, cmap='coolwarm', origin='lower')
    ax.set_title(f'Mean Attribute {i+1} per Neuron')
    fig.colorbar(im, ax=ax)

plt.tight_layout()
st.pyplot(fig)
