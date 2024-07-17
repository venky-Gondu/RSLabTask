import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def main():
    st.title("📊 K-means Clustering with PCA Visualization")
    
    st.markdown("""
    Welcome to the K-means Clustering app! This application allows you to upload an Excel file containing your dataset,
    perform K-means clustering on the data, and visualize the results using PCA (Principal Component Analysis).
    """)
    
    # File upload
    uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())
        
        # Check if 'target' column exists
        if 'target' in df.columns:
            # Prepare the data
            x = df.drop(['target'], axis=1)
            y = df['target']
            
            # Display DataFrame statistics
            st.markdown("### 📊 Data Description")
            st.dataframe(df.describe().T)
            
            # Scale the values between -1 and 1
            st.markdown("### 📈 Scaling Data")
            st.write("Scaling the feature values to be between -1 and 1 for better clustering results.")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
            
            # Define K-means clustering
            def apply_kmeans(data, n_clusters=6):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(data)
                return kmeans, cluster_labels
            
            # Apply K-means clustering
            st.markdown("### ⚙️ Applying K-means Clustering")
            st.write("Clustering the data into 6 clusters using the K-means algorithm.")
            kmeans, cluster_labels = apply_kmeans(df_scaled)
            
            # Apply PCA for dimensionality reduction
            st.markdown("### 🌐 PCA for Dimensionality Reduction")
            st.write("Reducing the data dimensions to 2 for visualization purposes.")
            pca = PCA(n_components=2)
            df_pca = pca.fit_transform(df_scaled)
            
            # Plot the clusters
            st.markdown("### 📉 K-means Clustering with PCA")
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=cluster_labels, palette='viridis', s=100)
            plt.title('K-means Clustering with PCA')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            st.pyplot(plt)
            
            # Predict the cluster of a data point
            st.markdown("### 🔮 Predicting the Cluster of the First Data Point")
            cluster = kmeans.predict([df_scaled.iloc[0].values])[0]
            st.write(f'The first data point is predicted to be in cluster **{cluster}**.')
            
            # Using distance to find the closest cluster
            st.markdown("### 📏 Calculating Distances to Cluster Centers")
            st.write("Calculating the distances from the first data point to each cluster center.")
            cluster_centers = kmeans.cluster_centers_
            distances = [np.linalg.norm(df_scaled.iloc[0].values - center) for center in cluster_centers]
            closest_cluster_index = np.argmin(distances)
            
            st.write(f"The first data point is closest to cluster **{closest_cluster_index}**.")
            
            # Display distances to each cluster center
            st.markdown("#### Distances to Each Cluster Center")
            for i, distance in enumerate(distances):
                st.write(f"Distance to cluster {i}: {distance:.4f}")
        
        else:
            st.error("The uploaded file does not contain a 'target' column. Please upload a file with the correct format.")
    else:
        st.info("Please upload an Excel file to proceed.")

if __name__ == "__main__":
    main()

