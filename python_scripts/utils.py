import numpy as np
from numba import jit
import pandas as pd
from torch.utils.data import Dataset


@jit(nopython=True)
def midi2frequency(
        midi: np.ndarray,
        base_frequency: float = 440.0,
) -> np.ndarray:
    """
    Convert MIDI note number to frequency.

    Args:
        midi (np.ndarray): The MIDI note number. Can be a scalar or an array.
        base_frequency (float, optional): The base frequency (or "tuning") to use. Defaults to 440.0.

    Returns:
        np.ndarray: The frequency in Hz.
    """
    return base_frequency * 2 ** ((midi.astype(np.float64) - 69) / 12)


@jit(nopython=True)
def frequency2midi(
        frequency: np.ndarray,
        base_frequency: float = 440.0,
) -> np.ndarray:
    """
    Converts a frequency in Hz to a MIDI note number.

    Args:
        frequency: Frequency in Hz. Can be a scalar or a numpy array.
        base_frequency: Frequency of MIDI note 69. Defaults to 440.0.

    Returns:
        np.ndarray: MIDI note number.
    """

    return 69 + 12 * np.log2(frequency.astype(np.float64) / base_frequency)


@jit(nopython=True)
def scale_array_auto(
    array: np.ndarray,
    out_low: float,
    out_high: float
) -> np.ndarray:
    """
    Scales an array linearly. The input range is automatically 
    retrieved from the array. Optimized by Numba.

    Args:
        array (np.ndarray): The array to be scaled.
        out_low (float): Minimum of output range.
        out_high (float): Maximum of output range.

    Returns:
        np.ndarray: The scaled array.
    """
    minimum, maximum = np.min(array), np.max(array)
    # if all values are the same, then return an array with the
    # same shape, all cells set to out_high
    if maximum - minimum == 0:
        return np.ones_like(array, dtype=np.float64) * out_high
    else:
        m = (out_high - out_low) / (maximum - minimum)
        b = out_low - m * minimum
        return m * array + b


@jit(nopython=True)
def resize_interp(
    input: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Resize an array. Uses linear interpolation.

    Args:
        input (np.ndarray): Array to resize.
        size (int): The new size of the array.

    Returns:
        np.ndarray: The resized array.
    """
    # create x axis for input
    input_x = np.arange(0, len(input))
    # create array with sampling indices
    output_x = scale_array_auto(np.arange(size), 0, len(input_x)-1)
    # interpolate
    return np.interp(output_x, input_x, input).astype(np.float64)


@jit(nopython=True)
def array2broadcastable(
    array: np.ndarray,
    samples: int
) -> np.ndarray:
    """
    Convert an array to a broadcastable array. If the array has a single value or has
    the size == samples, the array is returned. Otherwise the array is resized with 
    linear interpolation (calling resize_interp) to match the number of samples.

    Args:
        array (np.ndarray): The array to convert.
        samples (int): The number of samples to generate.

    Returns:
        np.ndarray: The converted array.
    """
    if array.size == 1 or array.size == samples:
        return array
    else:
        return resize_interp(array, samples)


@jit(nopython=True)
def wrap(
    x: float,
    min: float,
    max: float,
) -> float:
    """
    Wrap a value between a minimum and maximum value.

    Args:
        x (float): The value to wrap.
        min (float): The minimum value.
        max (float): The maximum value.

    Returns:
        float: The wrapped value.
    """
    return (x - min) % (max - min) + min


@jit(nopython=True)
def phasor(
    samples: int,
    sr: int,
    frequency: np.ndarray,
) -> np.ndarray:
    """
    Generate a phasor.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        frequency (np.ndarray): The frequency to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated phasor.
    """
    # create array to hold output
    output = np.zeros(samples, dtype=np.float64)
    frequency_resized = np.array([0], dtype=np.float64)
    if len(frequency) == 1:
        frequency_resized = np.repeat(frequency[0], samples).astype(np.float64)
    elif len(frequency) == samples:
        frequency_resized = frequency.astype(np.float64)
    else:
        # resize frequency array to match number of samples (-1 because we start at 0)
        frequency_resized = resize_interp(frequency, samples-1)
    # for each sample after the first
    for i in range(samples-1):
        # calculate increment
        increment = frequency_resized[i] / sr
        # calculate phasor value from last sample and increment
        output[i+1] = wrap(increment + output[i], 0, 1)
    return output


@jit(nopython=True)
def sinewave(
    samples: int,
    sr: int,
    frequency: np.ndarray,
) -> np.ndarray:
    """
    Generate a sine wave.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        frequency (np.ndarray): The frequency to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated sine wave.
    """
    # create phasor buffer
    phasor_buf = phasor(samples, sr, frequency)
    # calculate sine wave and return sine buffer
    return np.sin(2 * np.pi * phasor_buf)


def fm_synth_gen(
        samples: int,
        sr: int,
        carrier_frequency: np.ndarray,
        harmonicity_ratio: np.ndarray,
        modulation_index: np.ndarray,
) -> np.ndarray:
    """
    Generate a frequency modulated signal.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        carrier_frequency (np.ndarray): The carrier frequency to use. Can be a single value or an array.
        harmonicity_ratio (np.ndarray): The harmonicity ratio to use. Can be a single value or an array.
        modulation_index (np.ndarray): The modulation index to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated frequency modulated signal.
    """
    # initialize parameter arrays
    _carrier_frequency = array2broadcastable(
        carrier_frequency.astype(np.float64), samples)
    _harmonicity_ratio = array2broadcastable(
        harmonicity_ratio.astype(np.float64), samples)
    _modulation_index = array2broadcastable(
        modulation_index.astype(np.float64), samples)

    # calculate modulator frequency
    modulator_frequency = _carrier_frequency * _harmonicity_ratio
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    # create modulation amplitude buffer
    modulation_amplitude = modulator_frequency * _modulation_index
    # calculate frequency modulated signal and return fm buffer
    return sinewave(samples, sr, _carrier_frequency + (modulator_buf * modulation_amplitude))


class FmSynthDataset(Dataset):
    def __init__(self, csv_path, sr=48000, dur=1):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f_carrier = np.array([row.freq])
        harm_ratio = np.array([row.harm_ratio])
        mod_idx = np.array([row.mod_index])
        fm_synth = fm_synth_gen(int(self.dur * self.sr), self.sr,
                                f_carrier, harm_ratio, mod_idx)
        return fm_synth, row.freq, row.harm_ratio, row.mod_index


def array2fluid_dataset(
        array: np.ndarray,
) -> dict:
    """
    Convert a numpy array to a json format that's compatible with fluid.dataset~.

    Args:
        array (np.ndarray): The numpy array to convert. Should be a 2D array of (num_samples, num_features).

    Returns:
        dict: The json dataset.
    """
    num_cols = array.shape[1]
    out_dict = {}
    out_dict["cols"] = num_cols
    out_dict["data"] = {}
    for i in range(len(array)):
        out_dict["data"][str(i)] = array[i].tolist()
    return out_dict
