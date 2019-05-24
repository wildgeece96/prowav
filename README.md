# ProWav  
You can use this for preprocessing wave files.  

## Usage
Please install prowav by using pip.
```
pip install prowav  
```


### Usage 
ProWav can calculate mfcc spectrogram and pad for batch execution  
```python  
from prowav import ProWav  

prowav = ProWav(["path/to/wave/data_1.wav", "path/to/wave/data_2.wav"])  

frame_width = 20 # the length of a frame (ms)
stride_width = 20 # the frame interval (ms)
n_mfcc = 26 # the number of features by mfcc features (If you want to use mfcc preprocessing, you should specify this value)  
mode = 'MFCC'
window_func = 'hamming' # the name for window function
data = prowav.prepro(frame_width=20,stride_width=20,mode=mode,
                                       n_mfcc=n_mfcc, window_func=window_func)
# >> (num_files, num_frames, n_mfcc)    
```
If you want use fft spectrogram, please specify the mode, "fft".  
```python
frame_width = 20
stride_width = 20
mode='fft'
window_func='hamming'  
data = prowav.prepro(frame_width=20,stride_width=20,\
      mode=mode, window_func=window_func)  
# >> (num_files, num_frames, num_features)
```

Just loading wave data is possible.  

```python 
prowav = ProWav(["path/to/wave/data_1.wav", "path/to/wave/data_2.wav"]) 

prowav.load_wav() # loading wav file into this class.

prowav.data # the list of ndarray. Raw data are listed.
```
