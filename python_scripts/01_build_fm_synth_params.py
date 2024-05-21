# %%
# imports
import numpy as np
import pandas as pd
from utils import midi2frequency, array2fluid_dataset
import json

# %%
# create ranges for each parameter
pitch_steps = 51
harm_ratio_steps = 51
mod_idx_steps = 51
pitches = np.linspace(38, 86, pitch_steps)
freqs = midi2frequency(pitches)  # x
ratios = np.linspace(0, 1, harm_ratio_steps) * 10  # y
indices = np.linspace(0, 1, mod_idx_steps) * 10  # z
sr = 48000
dur = 1

# make into 3D mesh
freqs, ratios, indices = np.meshgrid(freqs, ratios, indices)  # y, x, z!
print(freqs.shape, ratios.shape, indices.shape)

# %%
# Create a dictionary where each key-value pair corresponds to a column in the dataframe
data = {
    "x": np.tile(np.repeat(np.arange(pitch_steps), mod_idx_steps), harm_ratio_steps),
    "y": np.repeat(np.arange(harm_ratio_steps), pitch_steps * mod_idx_steps),
    "z": np.tile(np.arange(mod_idx_steps), harm_ratio_steps * pitch_steps),
    "freq": freqs.flatten(),
    "harm_ratio": ratios.flatten(),
    "mod_index": indices.flatten()
}

# Create the dataframe
df = pd.DataFrame(data)
df.head()

# %%
# save to disk
df.to_csv("../data/fm_synth_params.csv", index=True)

# %%
# export fm params as fluid dataset
df_params_fm = df[["freq", "harm_ratio", "mod_index"]]
# convert to numpy array
df_params_fm = df_params_fm.values
# save as fluid dataset
df_params_fm_dict = array2fluid_dataset(df_params_fm)
# save to json
with open("../data/fm_params.json", "w") as f:
    json.dump(df_params_fm_dict, f)

# %%
# get scaled x y z for colors
x = df.x.values
y = df.y.values
z = df.z.values
# scale to between 0 and color_max
color_max = 0.9
x = (x - x.min()) / (x.max() - x.min()) * color_max
y = (y - y.min()) / (y.max() - y.min()) * color_max
z = (z - z.min()) / (z.max() - z.min()) * color_max
alpha = np.repeat(0.2, len(x))
colors = np.stack((x, y, z, alpha), axis=-1)

# %%
# save to disk
np.save("../data/colors.npy", colors)
# save the colors array
colors_dict = array2fluid_dataset(colors)
# save to json
with open("../data/colors.json", "w") as f:
    json.dump(colors_dict, f)

# %%
