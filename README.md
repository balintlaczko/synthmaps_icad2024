# Synth Maps
Code repository for my paper "Synth Maps: Mapping The Non-Proportional Relationships Between Synthesizer Parameters and Synthesized Sound" for ICAD2024.

Download the record from Zenodo to get all datasets (recommended):

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.11237788.svg)](http://dx.doi.org/10.5281/zenodo.11237788)

# Installation

## Install Python dependencies
To generate all datasets, you will need to install the following Python packages:
- numpy
- numba
- torch
- torchaudio
- pandas
- matplotlib
- tqdm
- timbral_models
- PyTimbre
- frechet_audio_distance

To install them via Pip, run the following command in your Terminal:

```bash
pip3 install numpy numba torch torchaudio pandas matplotlib tqdm timbral_models PyTimbre frechet_audio_distance
```

## Install Max
[Download](https://cycling74.com/downloads) and install Max. 

Then unzip *max/FluidCorpusManipulation.zip* (included in the Zenodo record) into *~/Documents/Max 8/Packages/*. This includes a [PR of the updated `fluid.jit.plotter` object](https://github.com/flucoma/flucoma-max/pull/417) that is not merged yet.

If the PR has been merged, you can also simply install the Fluid Corpus Manipulation package via Max's built-in Package Manager (*File > Show Package Manager*, then search for "Fluid Corpus Manipulation").

# Generate all datasets

After all dependencies have been installed, you are ready to generate the datasets used in this study. 
(This can take several hours on an average laptop, so if you are in a hurry, use the data included in the Zenodo record.)

First, open the Terminal and navigate to the *python_scripts* folder:
```bash
cd python_scripts
```

Then evaluate the six scripts in order to generate all datasets.

```bash
python3 01_build_fm_synth_params.py
```
```bash
python3 02_build_perceptual_ds.py
```
```bash
python3 03_build_fm_spectral_ds.py
```
```bash
python3 04_render_mel_spectrograms.py
```
```bash
python3 05_render_embeddings.py
```
```bash
python3 06_render_pca_plots.py
```

Note that *05_render_embeddings.py* will need internet connection to download the EnCodec and CLAP models.

# Interact with the data

When you generated all datasets (or using the data included in the Zenodo record), open Max, and open the *max/SynthMaps.maxpat* (from this repo). Then follow the instructions in the patch.

<img width="1402" alt="image" src="https://github.com/balintlaczko/synthmaps_icad2024/assets/45127463/698a0e2e-5396-4a65-851a-50b880e9c943">

