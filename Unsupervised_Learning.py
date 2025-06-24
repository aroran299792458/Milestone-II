#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from IPython.display import display


# Load merged CSV file
df = pd.read_csv("merged_data.csv", low_memory=False)

#We will only consider records where a delay due to weather occurred
df = df[df['WEATHER_DELAY'] > 0]


df['dep_hour'] = pd.to_datetime(df['dep_hour'], errors='coerce')

# Parsing functions for weather fields
# Metadata for this is available at: 
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
def parse_wnd(val):
    try:
        speed = int(str(val)[8:12])
        return None if speed == 9999 else speed / 10
    except:
        return None

def parse_cig(val):
    try:
        height = int(str(val)[:5])
        return None if height == 99999 else height
    except:
        return None

def parse_vis(val):
    try:
        distance = int(str(val)[:6])
        return None if distance == 999999 else distance
    except:
        return None

def parse_tmp(val):
    try:
        t = int(str(val)[:5])
        return None if t == 9999 else t / 10
    except:
        return None

def parse_aa1(val):
    try:
        v = str(val)
        numerator = int(v[3:7])
        if numerator == 9999:
            return None
        return (numerator / 10) 
    except:
        return None

def parse_aj1(val):
    try:
        depth = int(str(val)[:4])
        return None if depth == 9999 else depth
    except:
        return None

# Apply parsing
df['WindSpeed_mps'] = df['wnd'].apply(parse_wnd)
df['CloudBase_m'] = df['cig'].apply(parse_cig)
df['Visibility_m'] = df['vis'].apply(parse_vis)
df['Temperature_C'] = df['tmp'].apply(parse_tmp)
df['PrecipitationRate_mm_per_hr'] = df['aa1'].apply(parse_aa1)
df['SnowDepth_cm'] = df['aj1'].apply(parse_aj1)

# Decode aw1 codes
weather_code_map = {
    "00": "No significant weather observed", "01": "Clouds dissolving", "02": "Sky unchanged",
    "03": "Clouds developing", "04": "Haze/smoke/dust", "05": "Smoke", "07": "Dust/sand raised",
    "10": "Mist", "11": "Diamond dust", "12": "Distant lightning", "18": "Squalls", "20": "Fog",
    "21": "Precipitation", "22": "Drizzle/snow grains", "23": "Rain", "24": "Snow",
    "25": "Freezing drizzle/rain", "26": "Thunderstorm", "27": "Blowing/drifting snow/sand",
    "28": "Blowing/drifting snow/sand, vis ≥ 1km", "29": "Blowing/drifting snow/sand, vis < 1km",
    "30": "Fog", "31": "Fog/ice fog in patches", "32": "Fog/ice fog thinner", "33": "Fog/ice fog unchanged",
    "34": "Fog/ice fog thicker", "35": "Fog with rime", "40": "Precipitation",
    "41": "Precipitation slight/moderate", "42": "Precipitation heavy", "43": "Liquid precip. slight/mod.",
    "44": "Liquid precip. heavy", "45": "Solid precip. slight/mod.", "46": "Solid precip. heavy",
    "47": "Freezing precip. slight/mod.", "48": "Freezing precip. heavy", "50": "Drizzle",
    "51": "Drizzle slight", "52": "Drizzle moderate", "53": "Drizzle heavy",
    "54": "Freezing drizzle slight", "55": "Freezing drizzle moderate", "56": "Freezing drizzle heavy",
    "57": "Drizzle + rain slight", "58": "Drizzle + rain moderate/heavy", "60": "Rain",
    "61": "Rain slight", "62": "Rain moderate", "63": "Rain heavy", "64": "Freezing rain slight",
    "65": "Freezing rain moderate", "66": "Freezing rain heavy", "67": "Rain/drizzle + snow slight",
    "68": "Rain/drizzle + snow moderate/heavy", "70": "Snow", "71": "Snow slight", "72": "Snow moderate",
    "73": "Snow heavy", "74": "Ice pellets slight", "75": "Ice pellets moderate",
    "76": "Ice pellets heavy", "77": "Snow grains", "78": "Ice crystals", "80": "Showers",
    "81": "Rain showers slight", "82": "Rain showers moderate", "83": "Rain showers heavy",
    "84": "Rain showers violent", "85": "Snow showers slight", "86": "Snow showers moderate",
    "87": "Snow showers heavy", "89": "Hail", "90": "Thunderstorm", "91": "Thunderstorm no precip.",
    "92": "Thunderstorm + rain/snow", "93": "Thunderstorm + hail", "94": "Heavy thunderstorm no precip.",
    "95": "Heavy thunderstorm + rain/snow", "96": "Heavy thunderstorm + hail", "99": "Tornado"
}

def extract_weather_types(group):
    codes = set()
    for val in group.dropna():
        val = str(val)
        for i in range(0, len(val), 6):
            code = val[i:i+2]
            if code in weather_code_map:
                codes.add(weather_code_map[code])
    return "|".join(sorted(codes)) if codes else None

weather_types_by_hour = (df.groupby(["ORIGIN_CITY_NAME", "dep_hour"])["aw1"]
      .apply(extract_weather_types).reset_index())
weather_types_by_hour.rename(columns={"aw1": "Weather_Types"}, inplace=True)

# Define aggregation dictionary
agg_dict = {
    'YEAR': 'max', 'MONTH': 'max', 'DAY_OF_MONTH': 'max', 'DAY_OF_WEEK': 'max',
    'ORIGIN_AIRPORT_ID': 'max', 'ORIGIN_CITY_NAME': 'max', 'ORIGIN_STATE_ABR': 'max',
    'ORIGIN_STATE_NM': 'max', 'WEATHER_DELAY': 'mean', 'FL_DATE': 'max', 'station': 'max',
    'date': 'max', 'latitude': 'max', 'longitude': 'max', 'elevation': 'max', 'name': 'max',
    'WindSpeed_mps': 'mean', 'CloudBase_m': 'mean', 'Visibility_m': 'mean',
    'Temperature_C': 'mean', 'PrecipitationRate_mm_per_hr': 'mean', 'SnowDepth_cm': 'mean'
}

# Aggregate
df_grouped = df.groupby(["ORIGIN_CITY_NAME", "dep_hour"], as_index=False).agg(agg_dict)

# Merge in weather types
df_grouped = pd.merge(df_grouped, weather_types_by_hour, on=["ORIGIN_CITY_NAME", "dep_hour"], how="left")

# Fill missing values
df_grouped['SnowDepth_cm'] = df_grouped['SnowDepth_cm'].fillna(0)
df_grouped['PrecipitationRate_mm_per_hr'] = df_grouped['PrecipitationRate_mm_per_hr'].fillna(0)
df_grouped['Weather_Types'] = df_grouped['Weather_Types'].fillna("No significant weather observed")
df_grouped['WEATHER_DELAY'] = df_grouped['WEATHER_DELAY'].fillna(0)

# Save output
output_path = "Deduplicated_Weather_With_Types.csv"
df_grouped.to_csv(output_path, index=False)

display(df_grouped)




# In[2]:


# Reconstruct the data and continue since kernel state was lost
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
file_path = "Deduplicated_Weather_With_Types.csv"
df = pd.read_csv(file_path)

# Feature selection
features = [
    'WindSpeed_mps', 'CloudBase_m', 'Visibility_m', 'Temperature_C',
    'PrecipitationRate_mm_per_hr', 'SnowDepth_cm', 'WEATHER_DELAY'
]
X = df[features].dropna()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X_scaled)

# Prepare plotting DataFrame
df_plot = df.loc[X.index].copy()
df_plot["KMeans_Label"] = kmeans_labels
df_plot["DBSCAN_Label"] = dbscan_labels
df_plot["Agglo_Label"] = agglo_labels
df_plot["PCA1"] = X_pca[:, 0]
df_plot["PCA2"] = X_pca[:, 1]

# Function to plot PCA scatter with cluster labels
def plot_pca_clusters(data, label_col, title):
    plt.figure()
    sns.scatterplot(
        x="PCA1", y="PCA2",
        hue=label_col,
        data=data,
        palette="tab10",
        legend="full",
        s=60
    )
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

# PCA scatter plots
plot_pca_clusters(df_plot, "KMeans_Label", "KMeans Clusters (PCA Projection)")
plot_pca_clusters(df_plot, "DBSCAN_Label", "DBSCAN Clusters (PCA Projection)")
plot_pca_clusters(df_plot, "Agglo_Label", "Agglomerative Clusters (PCA Projection)")

# Cluster means (for radar chart) - use normalized values
cluster_means = {
    "KMeans": pd.DataFrame(X_scaled, columns=features).assign(Cluster=kmeans_labels).groupby("Cluster").mean(),
    "Agglomerative": pd.DataFrame(X_scaled, columns=features).assign(Cluster=agglo_labels).groupby("Cluster").mean()
}

display(cluster_means["KMeans"])
cluster_means["Agglomerative"]


# In[3]:


import numpy as np

# Radar chart generation for cluster characteristics (KMeans)
def plot_radar_chart(cluster_df, title):
    categories = list(cluster_df.columns)
    num_vars = len(categories)

    # Setup angles and figure
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for i, row in cluster_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_title(title, size=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

# Plot radar chart for KMeans
plot_radar_chart(cluster_means["KMeans"], "KMeans Cluster Characteristics (Normalized Features)")

# Create delay heatmap by airport
delay_by_airport = df_plot.groupby("ORIGIN_CITY_NAME")["WEATHER_DELAY"].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x="ORIGIN_CITY_NAME", y="WEATHER_DELAY", data=delay_by_airport)
plt.title("Average Weather Delay by Airport")
plt.xlabel("Airport City")
plt.ylabel("Avg Weather Delay (minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sensitivity analysis: KMeans with different k
k_values = range(2, 10)
s_scores = []
db_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    s_scores.append(s)
    db_scores.append(db)

# Plot sensitivity results
plt.figure(figsize=(10, 5))
plt.plot(k_values, s_scores, label="Silhouette Score", marker='o')
plt.plot(k_values, db_scores, label="Davies-Bouldin Score", marker='s')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")
plt.title("KMeans Sensitivity Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final summary table of best model from each family
summary = pd.DataFrame([
    {"Model": "KMeans (k=4)", "Silhouette": silhouette_score(X_scaled, kmeans_labels),
     "Davies-Bouldin": davies_bouldin_score(X_scaled, kmeans_labels)},
    {"Model": "DBSCAN (eps=1.5, min_samples=10)", "Silhouette": None if len(set(dbscan_labels)) <= 1
     else silhouette_score(X_scaled, dbscan_labels),
     "Davies-Bouldin": None if len(set(dbscan_labels)) <= 1
     else davies_bouldin_score(X_scaled, dbscan_labels)},
    {"Model": "Agglomerative (k=4)", "Silhouette": silhouette_score(X_scaled, agglo_labels),
     "Davies-Bouldin": davies_bouldin_score(X_scaled, agglo_labels)}
])
summary


# In[4]:


# Reload the re-uploaded file and re-execute clustering and visualizations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

# Load the dataset
df = pd.read_csv("Deduplicated_Weather_With_Types.csv")

# Define features and normalize
features = [
    'WindSpeed_mps', 'CloudBase_m', 'Visibility_m', 'Temperature_C',
    'PrecipitationRate_mm_per_hr', 'SnowDepth_cm', 'WEATHER_DELAY'
]
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Prepare dataframe for plotting
df_plot = df.loc[X.index].copy()
df_plot["KMeans_Label"] = kmeans_labels
df_plot["DBSCAN_Label"] = dbscan_labels
df_plot["Agglo_Label"] = agglo_labels
df_plot["PCA1"] = X_pca[:, 0]
df_plot["PCA2"] = X_pca[:, 1]

# Compute cluster means
kmeans_means = pd.DataFrame(X_scaled, columns=features).assign(Cluster=kmeans_labels).groupby("Cluster").mean()
agglo_means = pd.DataFrame(X_scaled, columns=features).assign(Cluster=agglo_labels).groupby("Cluster").mean()
dbscan_df = pd.DataFrame(X_scaled, columns=features).assign(Cluster=dbscan_labels)
dbscan_means = dbscan_df[dbscan_df["Cluster"] != -1].groupby("Cluster").mean()

# Define radar and heatmap functions
def plot_radar_chart(cluster_df, title):
    categories = list(cluster_df.columns)
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    for i, row in cluster_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_title(title, size=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

def plot_delay_heatmap(df_plot, label_col, title):
    heatmap_data = df_plot.groupby(["ORIGIN_CITY_NAME", label_col])["WEATHER_DELAY"].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="ORIGIN_CITY_NAME", columns=label_col, values="WEATHER_DELAY")
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Airport City")
    plt.tight_layout()
    plt.show()

# Radar Charts
plot_radar_chart(dbscan_means, "DBSCAN Cluster Characteristics (Normalized Features)")
plot_radar_chart(agglo_means, "Agglomerative Cluster Characteristics (Normalized Features)")

# Delay Heatmaps
plot_delay_heatmap(df_plot, "KMeans_Label", "KMeans: Avg Weather Delay by Airport and Cluster")
plot_delay_heatmap(df_plot, "DBSCAN_Label", "DBSCAN: Avg Weather Delay by Airport and Cluster (Excl. Noise)")
plot_delay_heatmap(df_plot, "Agglo_Label", "Agglomerative: Avg Weather Delay by Airport and Cluster")


# In[5]:


# Join real-world values for Agglomerative clustering (best Davies-Bouldin performance)
agglo_labels = agglo.fit_predict(X_scaled)
df_real = df.loc[X.index].copy()
df_real["Cluster"] = agglo_labels

# Compute real-world (non-normalized) means for each cluster
real_means = df_real.groupby("Cluster")[features].mean()

display(Markdown("## **Real-World Cluster Averages (Agglomerative Clustering, best performance)**"))
real_means


# In[6]:


# Reload the uploaded dataset and recompute DBSCAN and Agglomerative clusters
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from IPython.display import display, Markdown

# Load the dataset
df = pd.read_csv("Deduplicated_Weather_With_Types.csv")

# Feature preparation
features = [
    'WindSpeed_mps', 'CloudBase_m', 'Visibility_m', 'Temperature_C',
    'PrecipitationRate_mm_per_hr', 'SnowDepth_cm', 'WEATHER_DELAY'
]
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)


# Real-world means for DBSCAN (excluding noise)
df_dbscan = df.loc[X.index].copy()
df_dbscan["Cluster"] = dbscan_labels
dbscan_real_world = df_dbscan[df_dbscan["Cluster"] != -1].groupby("Cluster")[features].mean()

# Run KMeans with k=4
kmeans4 = KMeans(n_clusters=4, random_state=42)
kmeans4_labels = kmeans4.fit_predict(X_scaled)

# Assign cluster labels to the original DataFrame
df_kmeans4 = df.loc[X.index].copy()
df_kmeans4["Cluster"] = kmeans4_labels

# Compute real-world averages for each cluster
kmeans4_real_world = df_kmeans4.groupby("Cluster")[features].mean()

display(Markdown("## **Real-World Cluster Averages (KMeans, k=4)**"))
display(kmeans4_real_world)


display(Markdown("## **Real-World Cluster Averages (DBSCAN, best performance)**"))
dbscan_real_world


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=f5a99cd0-ffba-48ec-b167-e16ae5f7239b' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
