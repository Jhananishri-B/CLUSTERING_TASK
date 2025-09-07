import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("D:/AI WORKSHOP/TASK/KMEANS,HIERARCHIAL,DBSCAN/CC_GENERAL_preprocessed.csv")
features=['BALANCE','PURCHASES','CASH_ADVANCE','PAYMENTS','CREDIT_LIMIT',
          'PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','CASH_ADVANCE_FREQUENCY',
          'TENURE','PRC_FULL_PAYMENT']
df_features = df[features]

for col in ['PURCHASES','CASH_ADVANCE','PAYMENTS','CREDIT_LIMIT']:
    Q1 = df_features[col].quantile(0.25)
    Q3 = df_features[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df_features[col] = np.where(df_features[col] > upper_bound, upper_bound, df_features[col])

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df_features)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(pca_data, columns=["PC1","PC2"])

predefined_metrics = pd.DataFrame({
    "Model": ["K-Means", "Hierarchical", "DBSCAN"],
    "Silhouette": [0.436897, 0.544746, None],
    "Davies-Bouldin": [0.986939, 0.329792, None],
    "Calinski-Harabasz": [4289.943651, 7.417854, None],
    "Inertia": [71633.215304, None, None]
})

def run_clustering():
    results = []

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    pca_df['KMeans'] = kmeans_labels
    sil = silhouette_score(df_scaled, kmeans_labels)
    dbi = davies_bouldin_score(df_scaled, kmeans_labels)
    chi = calinski_harabasz_score(df_scaled, kmeans_labels)
    inertia = kmeans.inertia_
    results.append(["KMeans", sil, dbi, chi, inertia])

    hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hier_labels = hierarchical.fit_predict(df_scaled)
    pca_df['Hierarchical'] = hier_labels
    sil = silhouette_score(df_scaled, hier_labels)
    dbi = davies_bouldin_score(df_scaled, hier_labels)
    chi = calinski_harabasz_score(df_scaled, hier_labels)
    results.append(["Hierarchical", sil, dbi, chi, None])

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    db_labels = dbscan.fit_predict(df_scaled)
    pca_df['DBSCAN'] = db_labels
    if len(set(db_labels)) > 1 and -1 not in set(db_labels):
        sil = silhouette_score(df_scaled, db_labels)
        dbi = davies_bouldin_score(df_scaled, db_labels)
        chi = calinski_harabasz_score(df_scaled, db_labels)
    else:
        sil, dbi, chi = None, None, None
    results.append(["DBSCAN", sil, dbi, chi, None])

    results_df = pd.DataFrame(results, columns=["Model","Silhouette","Davies-Bouldin","Calinski-Harabasz","Inertia"])

    plots = []
    for col, title, palette in [
        ("KMeans","KMeans Clustering","tab10"),
        ("Hierarchical","Hierarchical Clustering","viridis"),
        ("DBSCAN","DBSCAN Clustering","plasma")
    ]:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x="PC1", y="PC2", hue=col, data=pca_df, palette=palette)
        plt.title(title, fontsize=14, color="blue")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="Cluster")
        plots.append(plt.gcf())
        plt.close()

    return results_df, plots[0], plots[1], plots[2]

def login(username, password):
    if username == "admin" and password == "1234":  
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True, value="❌ Wrong Username or Password"), gr.update(visible=False)

with gr.Blocks(theme="soft") as demo:
    with gr.Group(visible=True) as login_page:
        gr.HTML("<h1 style='text-align:center; color:#FF5733;'>🔑 Login Page</h1>")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        error_msg = gr.Textbox(label="Error", visible=False)
        login_btn = gr.Button("Login ✅")

    with gr.Group(visible=False) as main_page:
        gr.HTML("""
        <h1 style='text-align:center; color: blue; font-size: 40px;'>
        💳 Credit Card Customers Clustering App 
        </h1>
        <p style='text-align:center; color: green;'>
        This app allows you to cluster credit card customers using various algorithms.
        </p>
        """)
        
        gr.HTML("<h3 style='color:purple; text-align:center;'>📊 Predefined Evaluation Metrics</h3>")
        predefined_table = gr.Dataframe(value=predefined_metrics, interactive=False)
        
        run_btn = gr.Button("▶ Run Clustering")
        results_table = gr.Dataframe(headers=["Model","Silhouette","Davies-Bouldin","Calinski-Harabasz","Inertia"], label="📊 New Evaluation Metrics")
        kmeans_plot = gr.Plot(label="KMeans Visualization")
        hier_plot = gr.Plot(label="Hierarchical Visualization")
        dbscan_plot = gr.Plot(label="DBSCAN Visualization")

    login_btn.click(login, [username,password], [error_msg, main_page])
    run_btn.click(run_clustering, outputs=[results_table, kmeans_plot, hier_plot, dbscan_plot])

demo.launch()
