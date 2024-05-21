# %%
# imports
import numpy as np
import torch
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from utils import FmSynthDataset

# %%
# create the dataset
sr = 48000
dur = 1
csv_path = "../data/fm_synth_params.csv"
fm_synth_ds = FmSynthDataset(csv_path, sr=sr, dur=dur)

# %%
# render all mel spectrograms - mean
n_mels = 200
mel_spec = MelSpectrogram(
    sample_rate=48000,
    n_fft=4096,
    f_min=20,
    f_max=10000,
    pad=1,
    n_mels=n_mels,
    power=2,
    norm="slaney",
    mel_scale="slaney")

all_mel = np.zeros((len(fm_synth_ds), n_mels))

for i in tqdm(range(len(fm_synth_ds))):
    y, freq, ratio, index = fm_synth_ds[i]
    mel = mel_spec(torch.tensor(y, dtype=torch.float32))
    mel_avg = mel.mean(dim=1, keepdim=True)
    mel_avg_db = amplitude_to_DB(
        mel_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
    all_mel[i] = mel_avg_db.numpy().T

# %%
# save all_mel to disk - mean
print(all_mel.shape)
np.save("../data/fm_synth_mel_spectrograms_mean.npy", all_mel)
print("Saved fm_synth_mel_spectrograms_mean.npy")

# %%
