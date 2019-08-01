import numpy as np
import wavio
import librosa
import wave
from scipy import signal
from scipy import fromstring, int16
from scipy import fftpack
from tqdm import tqdm  
from joblib import Parallel, delayed 

from .utils import * 


class Sound(object):
    def __init__(self):
        self.data = None 
        self.wave_size = None 
        self.nchannel = None 
        self.sr = None 
        self.idx = None



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
    def _load_wav(self, file_path, idx, sr=None):
        result = Sound()
        result.idx = idx
        try:
            wave_file = wavio.read(file_path)
        except Exception as e:
            print(e)
            print("Cannot open %s" % file_path)
            raise ValueError
        wave_file.data = wave_file.data.astype(np.float32)
        if sr:
            wave_file.data = librosa.core.resample(wave_file.data.T, orig_sr=wave_file.rate, target_sr=sr, res_type="kaiser_fast").T
            result.sr = sr
        else:
            result.sr = wave_file.rate  
        wave_size = wave_file.data.shape[0]
        nchannel = wave_file.data.shape[1]
        result.wave_size = wave_size
        result.nchannel = nchannel  
        x = wave_file.data
        if nchannel == 2:
            x = x.mean(axis=1)  # streo to monoral
        else:
            x = x.flatten()
        result.data = x  
        return result 

    def load_wav(self, sr=None, verbose=0, parallel=False):
        """
        load wav files from self.file_paths
        sr : int. if you specify sr, you can get wave files which have a common samplerate.
        parallel : bool. You can do loading in parallel.
        """
        data = []
        
        if parallel:
            results = Parallel(n_jobs=-1, verbose=10*verbose,
                                backend='threading')([delayed(self._load_wav)(file_path,idx, sr) for idx, file_path in enumerate(self.file_paths)])
            data = [i for i in range(len(self.file_paths))]
            self.samplerates = [i for i in range(len(self.file_paths))]
            self.nchannels = [i for i in range(len(self.file_paths))]
            self.wave_sizes = [i for i in range(len(self.file_paths))]
            for result in results:
                self.samplerates[result.idx] = result.sr
                self.nchannels[result.idx] = result.idx 
                self.wave_sizes[result.idx] = result.wave_size                     
                data[result.idx] = result.data 
        else:
            if verbose==0:
                file_iter = self.file_paths 
            else:
                file_iter = tqdm(self.file_paths)
            results = [self._load_wav(file_path, idx, sr) for idx, file_path in enumerate(file_iter)]
            
            for result in results:
                self.samplerates.append(result.sr)
                self.nchannels.append(result.nchannel)
                self.wave_sizes.append(result.wave_size)
                data.append(result.data)  
        self.max_length = max(self.wave_sizes)  
        self.data = data

    def __prepro(self, i, x, sample_rate, wave_size, mode, frame_width, stride_width, n_mfcc, n_mels, window_func):
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
            x_spectrogram = mfcc(x_2d, window_func,n_mfcc=n_mfcc,sr=sample_rate)
            # >> (frame_num, num_per_frame)
        elif mode =='mel_spec':
            x_spectrogram = mel_spectrogram(x_2d, window_func, sr=sample_rate, n_mels=n_mels)
        else:
            raise ValueError("The mode %s is invalid"% mode)
        num_features = x_spectrogram.shape[1]
        return (i, x_spectrogram, num_features, frame_num)

    def _prepro(self, frame_width=20,stride_width=20,mode='fft',n_mfcc=None,window_func='boxcar', zero_padding=False,
                            repeat_padding=False, n_mels=30, parallel=False, sr=None,verbose=0):
        """
        inputs:
            frame_width : int. The length of frame for preprocessing (ms)
            stride_width: int. The hop size for preprocessing (ms)
            mode: {'fft', 'MFCC', 'mel_spec'}. Specify preprocessing way.
            zero_padding : bool. If return the value which padded with zero.
            repeat_padding : bool. Whether padding with parts of sequence.
            n_mfcc : int. Number of dimension for MFCC.
            n_mels : int. Number of dimension for mel-spectrogram.
            parallel : bool. Whether or not loading wave files in multi processes.
        returns:
            results: list of ndarray with shape (data_num, frame_num, num_per_frame).
             or if zero_padding is True, shape (data_num, max_frame_num, num_per_frame).
        """
        if not self.data:
            self.load_wav(parallel=parallel, sr=sr)
        results = [i for i in range(len(self.data))] # initialization
        self.num_features = [i for i in range(len(self.data))] # initialization 
        self.num_frames  = [i for i in range(len(self.data))] # initialization
        max_wave_size = max(self.wave_sizes)
        if parallel:
            result_list = Parallel(n_jobs=-1, verbose=verbose*10, backend='threading')([delayed(self.__prepro)(i=i,
                                                            x = self.data[i],
                                                            sample_rate=self.samplerates[i],
                                                            wave_size = self.wave_sizes[i],
                                                            mode=mode, 
                                                            frame_width=frame_width,
                                                            stride_width=stride_width,
                                                            n_mfcc=n_mfcc,
                                                            n_mels=n_mels,
                                                            window_func=window_func,
                                                            ) for i in range(len(self.data))
                                                        ])  
            for i in range(len(self.data)):
                result = result_list[i]
                results[result[0]] = result[1]
                self.num_features[result[0]] = result[2]
                self.num_frames[result[0]] = result[3]
        else:
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
                results[i] = x_spectrogram
                self.num_features[i] = x_spectrogram.shape[1]
                self.num_frames[i] = frame_num
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
                repeat_padding=False, n_mels=None, parallel=False, sr=None, verbose=0):
        """
        inputs:
            frame_width : int. The length of frame for preprocessing (ms)
            stride_width: int. The hop size for preprocessing (ms)
            mode: {'fft', 'MFCC', 'mel_spec'}. Specify preprocessing way.
            zero_padding : bool. If return the value which padded with zero.
            repeat_padding : bool. Whether padding with parts of sequence.
            n_mfcc : int. Number of dimension for MFCC.
            n_mels : int. Number of dimension for mel-spectrogram.
            parallel : bool. Whether or not loading wave files in multi processes.
            sr : int. You can specify samperate when loading wav file.
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
                        repeat_padding=repeat_padding, n_mels=n_mels, parallel=parallel,sr=sr, verbose=verbose)
        return results


