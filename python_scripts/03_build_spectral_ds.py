from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pytimbre.waveform import Waveform
from pytimbre.spectral.spectra import SpectrumByFFT
import pandas as pd
from utils import FmSynthDataset
import json

# create the dataset
sr = 48000
dur = 1
csv_path = "../data/fm_synth_params.csv"
fm_synth_ds = FmSynthDataset(csv_path, sr=sr, dur=dur)


def extract_features(i, synths, sr):
    y, freq, ratio, index = synths[i]
    wfm = Waveform(y, sr, 0.0)
    spectrum = SpectrumByFFT(wfm, 4096)
    timbre = {
        "index": i,
        "freq": freq,
        "harm_ratio": ratio,
        "mod_index": index,
        "spectral_centroid": spectrum.spectral_centroid,
        "spectral_crest": spectrum.spectral_crest,
        "spectral_decrease": spectrum.spectral_decrease,
        "spectral_energy": spectrum.spectral_energy,
        "spectral_flatness": spectrum.spectral_flatness,
        "spectral_kurtosis": spectrum.spectral_kurtosis,
        "spectral_roll_off": spectrum.spectral_roll_off,
        "spectral_skewness": spectrum.spectral_skewness,
        "spectral_slope": spectrum.spectral_slope,
        "spectral_spread": spectrum.spectral_spread,
        "inharmonicity": spectrum.inharmonicity
    }
    return timbre


if __name__ == '__main__':
    executor = ProcessPoolExecutor()
    jobs = [executor.submit(extract_features, idx, fm_synth_ds, sr)
            for idx in range(len(fm_synth_ds))]
    results = []
    for job in tqdm(as_completed(jobs), total=len(jobs)):
        results.append(job.result())
    print("Finished extracting features")

    # save to json
    json_object = json.dumps(results, indent=4)
    outfile_path = "../data/fm_synth_spectral_features.json"
    with open(outfile_path, "w") as outfile:
        outfile.write(json_object)
    print("Features saved to json")

    # save as csv
    # read perceptual features from json
    with open(outfile_path, "r") as f:
        data = json.load(f)
    df_perceptual = pd.DataFrame(data)
    df_perceptual.set_index("index", inplace=True)
    # order by index
    df_perceptual.sort_index(inplace=True)
    df_perceptual.to_csv(
        "../data/fm_synth_spectral_features.csv", index=True)
    print("Features saved to csv")
