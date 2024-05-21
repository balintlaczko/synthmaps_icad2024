# %%
# imports
import numpy as np
from utils import FmSynthDataset
from frechet_audio_distance import FrechetAudioDistance
from tqdm import tqdm

# %%
# create the dataset
sr = 48000
dur = 0.25  # we use 0.25 seconds for the embeddings
csv_path = "../data/fm_synth_params.csv"
fm_synth_ds = FmSynthDataset(csv_path, sr=sr, dur=dur)
test_fm = fm_synth_ds[0][0]

# %%
# render all synths
all_y = np.zeros((len(fm_synth_ds), test_fm.shape[0]))
for i in tqdm(range(len(fm_synth_ds))):
    y, freq, ratio, index = fm_synth_ds[i]
    all_y[i] = y

# %%
# use encodec
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/encodec",
    model_name="encodec",
    sample_rate=48000,
    channels=2,
    verbose=False
)

# %%
# iterate all_y one by one and render embeddings - ENCODEC
test_embs = frechet.get_embeddings([test_fm], sr)
all_embs = np.zeros((len(fm_synth_ds), test_embs.shape[0], test_embs.shape[1]))
for i in tqdm(range(len(fm_synth_ds))):
    synth = all_y[i]
    embs = frechet.get_embeddings([synth], sr)
    all_embs[i] = embs

# %%
# save all_embs to disk - ENCODEC
print(all_embs.shape)
np.save("../data/fm_synth_encodec_embeddings.npy", all_embs)
print("Saved fm_synth_encodec_embeddings.npy")

# %%
# use clap
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/clap",
    model_name="clap",
    sample_rate=48000,
    verbose=False
)

# %%
# iterate all_y one by one and render embeddings - CLAP
test_embs = frechet.get_embeddings([test_fm], sr)
all_embs = np.zeros((len(fm_synth_ds), test_embs.shape[-1]))
for i in tqdm(range(len(fm_synth_ds))):
    synth = all_y[i]
    embs = frechet.get_embeddings([synth], sr)
    all_embs[i] = embs

# %%
# save all_embs to disk - CLAP
print(all_embs.shape)
np.save("../data/fm_synth_clap_embeddings.npy", all_embs)
print("Saved fm_synth_clap_embeddings.npy")
