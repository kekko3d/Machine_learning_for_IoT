import os
import sounddevice as sd
from time import time
from scipy.io.wavfile import write
import tensorflow as tf
import tensorflow_io as tfio
import sys

#----------------------------------------FUNCTIONS----------------------------------------------------

def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1
    indata = tf.squeeze(indata)
    return indata


def get_spectrogram(array_audio, downsampling_rate, frame_length_in_s, frame_step_in_s, sampling_rate):
    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        array_audio = tfio.audio.resample(array_audio, sampling_rate_int64, downsampling_rate) 

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    stft = tf.signal.stft(
        array_audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )

    spectrogram = tf.abs(stft)
    return spectrogram, downsampling_rate


def is_silence(array_audio, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres, sampling_rate):
    spectrogram, sampling_rate = get_spectrogram(
        array_audio,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s,
        sampling_rate   
    )

    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s
    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1

    
def callback(indata, frames, callback_time, status):
    downsampling_rate = 16000
    frame_length_in_s = 0.064
    dbFSthres = -120
    duration_thres = 0.1

    array_audio = get_audio_from_numpy(indata)
    silence = is_silence(array_audio, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres, 16000)
    if silence == 0:
        timestamp = time()
        write(f'{timestamp}.wav', sampling_rate, indata)

        
#----------------------------------------END FUNCTIONS----------------------------------------------------
        

sampling_rate = 16000


if len(sys.argv) != 3:
    print("\'--device\' followed by channel id must be entered")
    sys.exit()
if sys.argv[1] != "--device":
    print("the only variables accepted is \'--device\' followed by channel id")
    sys.exit()
try:
    device = int(sys.argv[2])
except ValueError:
        print("channel id must be an integer")
        sys.exit()



with sd.InputStream(device=device, channels=1, dtype='int16', samplerate=sampling_rate, blocksize=16000, callback=callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break