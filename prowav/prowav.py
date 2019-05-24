import numpy as np
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
        self.framerates = []
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
                wave_file = wave.open(file_path, 'r')
            except Exception as e:
                print(e)
                print("Cannot open %s" % file_path)
            wave_size = wave_file.getnframes()
            if self.max_length < wave_size:
                self.max_length = wave_size
            nchannel = wave_file.getnchannels()
            self.nchannels.append(nchannel)
            self.wave_sizes.append(wave_size)
            self.framerates.append(wave_file.getframerate())
            self.samplerates.append(wave_file.getframerate())
            x = wave_file.readframes(wave_size)
            x = np.frombuffer(x, dtype='int16')
            if nchannel == 2:
                x = x[0, :]  # streo to monoral
            data.append(x)
        self.data = data

    def _prepro(self, frame_width=20,stride_width=20,mode='fft',n_mfcc=None,window_func=None):
        """
        inputs:
            frame_width : int. The length of frame for preprocessing (ms)
            stride_width: int. The hop size for preprocessing (ms)
            mode: {'fft', 'MFCC'}. Specify preprocessing way.
        returns:
            results: ndarray with shape (frame_num, num_per_frame).
        """
        if not self.data:
            self.load_wav()
        results = []
        for i in range(len(self.data)):
            x = self.data[i]
            sample_rate = self.framerates[i]
            wave_size = self.wave_sizes[i]

            num_per_frame = int(frame_width /1000 * sample_rate)
            stride_per_frame = int(stride_width / 1000 * sample_rate)
            wave_length = int(wave_size - wave_size%num_per_frame)

            frame_num = int((wave_size-num_per_frame)//stride_per_frame) # number of frame
            if not frame_num:
                raise ValueError
            x_2d = np.zeros((frame_num, num_per_frame), dtype='int16')
            for j in range(frame_num):
                x_2d[j] = x[ j*stride_per_frame: j*stride_per_frame + num_per_frame]
            if mode == 'fft':
                if not window_func:
                    x_2d = window(x_2d, window_func)
                x_spectrogram = np.fft.fft(x_2d) # >> (frame_num, num_per_frame)
                x_spectrogram = np.abs(x_spectrogram)
            elif mode=='MFCC':
                x_spectrogram = mfcc(x_2d, window,n_mfcc=n_mfcc,sr=self.samplerates[i])
                 # >> (frame_num, num_per_frame)
            else:
                raise ValueError("The mode %s is invalid"% mode)
            results.append(x_spectrogram)
            self.num_features.append(x_spectrogram.shape[1] )
            self.num_frames.append(frame_num)
        return results

    def prepro(self, mode='fft', frame_width=20, stride_width=20,n_mfcc=None,window_func=None):
        """
        return :
         results: list of ndarray. List of  preprocessed data which has shape (frame_num, num_per_frame)
        """
        if mode=='MFCC' and not n_mfcc:
            raise ValueError("n_mfcc should be specified if you choose mode MFCC")
        results = self._prepro(frame_width=frame_width,stride_width=stride_width,mode=mode,n_mfcc=n_mfcc,window_func=window_func)
        return results




def window(x_2d, window):
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
    window_func = signal.windows.get_windows(window, N)
    result = x_2d * window
    return result

def mfcc(x_2d, window,n_mfcc=26, sr=16000):
    emphasis_signal = preEmphasis(x_2d, 0.97)
    if not window:
        emphasis_signal = window(emphasis_signal, window)
    spec = np.abs(np.fft.fft(emphasis_signal, axis=-1))
    nfft = spec.shape[-1]
    spec = spec[:, :nfft//2+1]
    melfilters = librosa.filters.mel(sr=sr,n_fft=nfft,fmax=sr//2, n_mels=n_mfcc)
    mspec = np.log10(np.dot(spec, melfilters.T)+1e-10)
    ceps = fftpack.dct(mspec,type=2, norm='ortho',axis=-1)
    return ceps

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
        
