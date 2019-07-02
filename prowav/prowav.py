import numpy as np
import wavio
import librosa
import wave
from scipy import signal
from scipy import fromstring, int16
from scipy import fftpack
class ProWav(object):
    def __init__(self, file_paths=None):
        """
        input:
          file_paths : list. list of file paths
          check_file: bool. If True, the length of data will be checked.
                                        When the length of data is not uniform, the data with not uniform length will be excluded.
        """
        self.file_paths = file_paths
        self.wave_sizes = []
        self.nchannels = []
        self.num_features = []
        self.num_frames = []
        self.samplerates = []
        self.data = None
        self.max_length = 0


    def check_file(self):
        valid_file_list = []
        valid_length = []
        invalid_file_list = []
        for file_path in self.file_paths:
            try:
                wave_file = wave.open(file_path, 'r')
                valid_file_list.append(file_path)
            except Exception as e:
                print(e)
                print("Cannot open %s" % file_path)
                invalid_faile_list.append(file_path)
            valid_length.append(wave_file.getnframes())
            wave_file.close()
    def load_wav(self):
        """
        load wav files from self.file_paths
        """
        data = []
        for file_path in self.file_paths:
            try:
                wave_file = wavio.read(file_path)
            except Exception as e:
                print(e)
                print("Cannot open %s" % file_path)
                raise ValueError
            wave_size = wave_file.data.shape[0]
            if self.max_length < wave_size:
                self.max_length = wave_size
            nchannel = wave_file.data.shape[1]
            self.nchannels.append(nchannel)
            self.wave_sizes.append(wave_size)
            self.samplerates.append(wave_file.rate)
            x = wave_file.data
            if nchannel == 2:
                x = x.mean(axis=1)  # streo to monoral
            data.append(x)
        self.data = data

    def _prepro(self, frame_width=20,stride_width=20,mode='fft',n_mfcc=None,window_func='boxcar', zero_padding=False,
                            repeat_padding=False, n_mels=30):
        """
        inputs:
            frame_width : int. The length of frame for preprocessing (ms)
            stride_width: int. The hop size for preprocessing (ms)
            mode: {'fft', 'MFCC', 'mel_spec'}. Specify preprocessing way.
            zero_padding : bool. If return the value which padded with zero.
            repeat_padding : bool. Whether padding with parts of sequence.
        returns:
            results: list of ndarray with shape (data_num, frame_num, num_per_frame).
             or if zero_padding is True, shape (data_num, max_frame_num, num_per_frame).
        """
        if not self.data:
            self.load_wav()
        results = []
        self.num_features = []
        self.num_frames  = []
        max_wave_size = max(self.wave_sizes)
        for i in range(len(self.data)):
            x = self.data[i]
            sample_rate = self.samplerates[i]
            wave_size = self.wave_sizes[i]

            num_per_frame = int(frame_width /1000 * sample_rate)
            stride_per_frame = int(stride_width / 1000 * sample_rate)

            frame_num = int((wave_size-num_per_frame)//stride_per_frame) # number of frame
            if not frame_num:
                raise ValueError
            x_2d = np.zeros([frame_num, num_per_frame], dtype=x.dtype)
            for j in range(frame_num):
                x_2d[j] = x[j*stride_per_frame:j*stride_per_frame+num_per_frame] 
            if mode == 'fft':
                if not window_func:
                    x_2d = window_(x_2d, window_func)
                x_spectrogram = np.fft.fft(x_2d) # >> (frame_num, num_per_frame)
                x_spectrogram = np.abs(x_spectrogram)
            elif mode=='MFCC':
                x_spectrogram = mfcc(x_2d, window_func,n_mfcc=n_mfcc,sr=self.samplerates[i])
                 # >> (frame_num, num_per_frame)
            elif mode =='mel_spec':
                x_spectrogram = mel_spectrogram(x_2d, window_func, sr=self.samplerates[i], n_mels=n_mels)
            else:
                raise ValueError("The mode %s is invalid"% mode)
            results.append(x_spectrogram)
            self.num_features.append(x_spectrogram.shape[1] )
            self.num_frames.append(frame_num)
        if zero_padding:
            max_frame_num = max(self.num_frames)
            results_ = np.zeros([len(self.data), max_frame_num, results[0].shape[1]], dtype=np.float)
            for i in range(len(self.data)):
                seq_len = self.num_frames[i]
                results_[i, :seq_len] = results[i]
            return results_
        elif repeat_padding:
            max_frame_num = max(self.num_frames)
            results_ = make_batch_uniform_length(results, seq_len=max_frame_num, ds_rate=1,
                                            num_features=results[0].shape[-1])  
            return results_

        return results

    def prepro(self, mode='fft', frame_width=20, stride_width=20,n_mfcc=None,window_func='boxcar', zero_padding=False,
                repeat_padding=False, n_mels=None):
        """
        return :
         results: list of ndarray. List of  preprocessed data which has shape (frame_num, num_per_frame)
        """
        if mode=='MFCC' and not n_mfcc:
            raise ValueError("n_mfcc should be specified if you choose mode MFCC")
        if zero_padding == repeat_padding and zero_padding==True:
            raise ValueError("You can not choose two padding mode. Please choose only one. repeat_padding or zero_padding")
        if mode=='mel_spec' and not n_mels:
            raise ValueError("n_mels should be specified if you choose mode mel_spec")
        results = self._prepro(frame_width=frame_width,stride_width=stride_width,mode=mode,n_mfcc=n_mfcc,window_func=window_func,zero_padding=zero_padding,
                        repeat_padding=repeat_padding, n_mels=n_mels)
        return results




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