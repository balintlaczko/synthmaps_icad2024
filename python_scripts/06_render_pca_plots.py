# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils import frequency2midi, array2fluid_dataset
import json

# %%
# create a 2D scatter plot of the PCA-d synth parameters

# read dataset
df_params = pd.read_csv("../data/fm_synth_params.csv", index_col=0)
# get the freq column
freq = df_params["freq"].values
# translate to midi
midi = frequency2midi(freq)
# add pitch column
df_params["pitch"] = midi
# extract pitch, harm_ratio, mod_index
df_params_3d = df_params[["pitch", "harm_ratio", "mod_index"]]

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_params_3d)
# transform
df_params_3d = scaler.transform(df_params_3d)

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_params_3d)
# transform
df_params_2d = pca.transform(df_params_3d)

# get scaled x y z for colors
x = df_params.x.values
y = df_params.y.values
z = df_params.z.values
# scale to between 0 and color_max
color_max = 0.9
x = (x - x.min()) / (x.max() - x.min()) * color_max
y = (y - y.min()) / (y.max() - y.min()) * color_max
z = (z - z.min()) / (z.max() - z.min()) * color_max
alpha = np.repeat(0.2, len(x))
colors = np.stack((x, y, z, alpha), axis=-1)

# create scatter plot with small dots and color by x, y, z as RGB
plt.figure(dpi=300)
plt.scatter(df_params_2d[:, 0], df_params_2d[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth parameters\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_params.png", format="png")

# save the colors array
colors_dict = array2fluid_dataset(colors)
with open("../data/colors.json", "w") as f:
    json.dump(colors_dict, f)
# save the pca plot as a fluid dataset
pca_params = array2fluid_dataset(df_params_2d)
with open("../data/pca_params.json", "w") as f:
    json.dump(pca_params, f)


# %%
# create a 2D scatter plot of the PCA-d perceptual features

# read dataset
df_perceptual = pd.read_csv(
    "../data/fm_synth_perceptual_features.csv", index_col=0)
# extract perceptual features
df_perceptual_7d = df_perceptual[[
    "hardness", "depth", "brightness", "roughness", "warmth", "sharpness", "boominess"]]

# replace inf values to the next highest in column
df_perceptual_7d_filtered = df_perceptual_7d.replace([np.inf, -np.inf], np.nan)
# get list of columns
cols = df_perceptual_7d_filtered.columns
# iterate through columns
for col in cols:
    # get the max value
    max_val = df_perceptual_7d_filtered[col].max()
    # replace nan values with max value
    df_perceptual_7d_filtered[col] = df_perceptual_7d_filtered[col].fillna(
        max_val)

# clip all columns to between nth and (1-n)th percentile
n = 0.1
for col in df_perceptual_7d_filtered.columns:
    # get 10th and 90th percentile
    low = df_perceptual_7d_filtered[col].quantile(n)
    high = df_perceptual_7d_filtered[col].quantile(1-n)
    # clip
    df_perceptual_7d_filtered.loc[:, col] = df_perceptual_7d_filtered[col].clip(
        low, high)

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_perceptual_7d_filtered)
# transform
df_perceptual_7d_filtered_scaled = scaler.transform(df_perceptual_7d_filtered)
df_perceptual_7d_filtered_scaled = pd.DataFrame(
    df_perceptual_7d_filtered_scaled, columns=cols)

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_perceptual_7d_filtered_scaled)
# transform
df_perceptual_2d = pca.transform(df_perceptual_7d_filtered_scaled)

# create scatter plot with small dots
plt.figure(dpi=300)
plt.scatter(df_perceptual_2d[:, 0], df_perceptual_2d[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth perceptual features\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_perceptual.png", format="png")

# save the pca plot as a fluid dataset
pca_perceptual = array2fluid_dataset(df_perceptual_2d)
with open("../data/pca_perceptual.json", "w") as f:
    json.dump(pca_perceptual, f)


# %%
# create a 2D scatter plot of the PCA-d spectral features

# read dataset
df_spectral = pd.read_csv(
    "../data/fm_synth_spectral_features.csv", index_col=0)

# extract spectral features
df_spectral_11d = df_spectral[[
    "spectral_centroid", "spectral_crest", "spectral_decrease", "spectral_energy", "spectral_flatness", "spectral_kurtosis", "spectral_roll_off", "spectral_skewness", "spectral_slope", "spectral_spread", "inharmonicity"]]

# translate spectral_roll_off, spectral_centroid, spectral_spread to midi (making a linear scale out of an exponential one)
for col in ["spectral_roll_off", "spectral_centroid", "spectral_spread"]:
    # get the column
    values = df_spectral_11d[col].values
    # translate to midi
    midi = frequency2midi(values)
    # replace values
    df_spectral_11d.loc[:, col] = midi

# clip all columns to between nth and (1-n)th percentile
n = 0.1
for col in df_spectral_11d.columns:
    # get 10th and 90th percentile
    low = df_spectral_11d[col].quantile(n)
    high = df_spectral_11d[col].quantile(1-n)
    # clip
    df_spectral_11d.loc[:, col] = df_spectral_11d[col].clip(low, high)

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_spectral_11d)
# transform
df_spectral_11d_scaled = scaler.transform(df_spectral_11d)
df_spectral_11d_scaled = pd.DataFrame(
    df_spectral_11d_scaled, columns=df_spectral_11d.columns)

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_spectral_11d_scaled)
# transform
df_spectral_2d = pca.transform(df_spectral_11d_scaled)

# create scatter plot with small dots
plt.figure(dpi=300)
plt.scatter(df_spectral_2d[:, 0], df_spectral_2d[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth spectral features\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_spectral.png", format="png")

# save the pca plot as a fluid dataset
pca_spectral = array2fluid_dataset(df_spectral_2d)
with open("../data/pca_spectral.json", "w") as f:
    json.dump(pca_spectral, f)


# %%
# create pca plot for embeddings - ENCODEC
# read embeddings
embeddings = np.load("../data/fm_synth_encodec_embeddings.npy")
embeddings_2d = embeddings.reshape((embeddings.shape[0], -1))

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(embeddings_2d)
# transform
embeddings_2d_pca = pca.transform(embeddings_2d)

# create scatter plot with small dots
plt.figure(dpi=300)
plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth EnCodec embeddings\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_encodec.png", format="png")

# save the pca plot as a fluid dataset
pca_embeddings = array2fluid_dataset(embeddings_2d_pca)
with open("../data/pca_encodec.json", "w") as f:
    json.dump(pca_embeddings, f)

# %%
# create pca plot for embeddings - CLAP
# read embeddings
embeddings = np.load("../data/fm_synth_clap_embeddings.npy")

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(embeddings)
# transform
embeddings_2d_pca = pca.transform(embeddings)

# create scatter plot with small dots
plt.figure(dpi=300)
plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth CLAP embeddings\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_clap.png", format="png")

# save the pca plot as a fluid dataset
pca_embeddings = array2fluid_dataset(embeddings_2d_pca)
with open("../data/pca_clap.json", "w") as f:
    json.dump(pca_embeddings, f)


# %%
# create pca plot for mel spectrograms - mean
# read mel spectrograms
mel_spectrograms = np.load("../data/fm_synth_mel_spectrograms_mean.npy")

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(mel_spectrograms)
# transform
mel_spectrograms_2d_pca = pca.transform(mel_spectrograms)

# create scatter plot with small dots
plt.figure(dpi=300)
plt.scatter(mel_spectrograms_2d_pca[:, 0],
            mel_spectrograms_2d_pca[:, 1], s=1, c=colors)
fontsize = 18
plt.xlabel("PCA – 1st component", fontsize=fontsize)
plt.ylabel("PCA – 2nd component", fontsize=fontsize)
plt.title(
    f"PCA of FM synth mel spectrograms (mean)\nexplained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}", fontsize=fontsize)

# save figure as png
plt.savefig("../figures/pca_mels_mean.png", format="png")

# save the pca plot as a fluid dataset
pca_mels = array2fluid_dataset(mel_spectrograms_2d_pca)
with open("../data/pca_mels_mean.json", "w") as f:
    json.dump(pca_mels, f)

# %%
