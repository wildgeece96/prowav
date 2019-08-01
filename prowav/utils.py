import numpy as np
import wavio
import librosa
import wave
from scipy import signal
from scipy import fromstring, int16
from scipy import fftpack




def window_(x_2d, window):
    """
    inputs:
        x_2d : inputs (2D array)
        window : string, float, or tuple
                        The type of window to create. See below for more details.
    outputs:
        results: The ndarray that window function was applied to.

    We use window function from scipy.signal.windows.sget_window.
    For more details,  please see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
    Window types:
    `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
    `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,
    `barthann`, `kaiser` (needs beta), `gaussian` (needs standard
    deviation), `general_gaussian` (needs power, width), `slepian`
    (needs width), `dpss` (needs normalized half-bandwidth),
    `chebwin` (needs attenuation), `exponential` (needs decay scale),
    `tukey` (needs taper fraction)
    """
    N = x_2d.shape[1]
    window_func = signal.get_window(window, N)
    result = x_2d * window_func
    return result

def mfcc(x_2d, window_func,n_mfcc=26, sr=16000):
    mspec = mel_spectrogram(x_2d, window_func, n_mels=n_mfcc*4, sr=sr)
    ceps = fftpack.dct(mspec,type=2, norm='ortho',axis=-1)
    return ceps[:, 1:n_mfcc+1]

def mel_spectrogram(x_2d, window_func, sr=16000, n_mels=30):
    emphasis_signal = preEmphasis(x_2d, 0.97)
    if window_func != None:
        emphasis_signal = window_(emphasis_signal, window_func) 
    spec = np.abs(np.fft.fft(emphasis_signal, axis=-1))
    n_fft = spec.shape[-1]
    spec = spec[:, :n_fft//2+1]
    melfilters = librosa.filters.mel(sr=sr, n_fft=n_fft, fmax=sr//2, n_mels=n_mels)
    mspec = np.log10(np.dot(spec, melfilters.T)+1e-10)
    return mspec 
def preEmphasis(x, p):
    """
    input:
        x: 2d-array
        p : coefficient for pre emphasis
    output:
        result: 2d-array. preEmphasized data
    """
    result = np.zeros_like(x)
    for i in range(result.shape[0]):
        result[i] = np.convolve(x[i], [1, -p], mode='same')
    return result
        
def fix_audio_length(data, seq_len=2000, ds_rate=1):
    if len(data)>seq_len:
        return data[:seq_len]
    else:
        return repeat_audio_length(data, seq_len, ds_rate=1)
def repeat_audio_length(data, seq_len, ds_rate=1):
    if len(data.shape) == 2:
        results = np.zeros([seq_len, data.shape[1]], dtype=np.int16)
    elif len(data.shape)==1:
        results = np.zeros(seq_len, dtype=np.int16)
    else:
        raise ValueError(f"Invalid shape of data which has {data.shape}")
    data_len = data.shape[0]//ds_rate
    data = data[::ds_rate]
    step = 0
    while (step+1)*data_len < seq_len:
        results[step*data_len:(step+1)*data_len] = data
        step += 1 
    results[step*data_len:] = data[:seq_len-step*data_len]
    return results 

def make_batch_uniform_length(datas, seq_len=2000, ds_rate=1, num_features=0):
    """
    make batch which has uniform length of data.
    inputs: 
        datas : list or array of data. data is 2d or 1d array.
        seq_len : int. The max number of frame in data. 
        ds_rate : int. If you need to downsapling, you can choose downsampling rate.  
        num_features : int. The num of features per frame.
    returns:
        datas : array of data. data is 2d (seq_len, num_features), 1d(seq_len) array.

    """
    n_data = len(datas)
    if num_features != 0:
        X = np.zeros([n_data, seq_len, num_features], dtype=datas[0].dtype)
    else:
        X = np.zeros([n_data, seq_len], dtype=datas[0].dtype)
    for i in range(n_data):
        X[i] = fix_audio_length(datas[i], seq_len, ds_rate=ds_rate)
    return X 