import argparse
import psutil as psu
import redis
import uuid
import sounddevice as sd
from time import time
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow.lite as tflite
import numpy as np

MEL_LOG_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.05,
    'frame_step_in_s': 0.025,
    'num_mel_bins': 30,    
    'lower_frequency': 20,
    'upper_frequency': 8000,
}

CALLBACK_ARGS = {
    'downsampling_rate' : 16000,
    'frame_length_in_s' : 0.05,
    'frame_step_in_s': 0.025,
    'dbFSthres' : -120,
    'duration_thres' : 0.1,
}

SAMPLING_RATE = 16000


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--device', type=int)
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)

    args = parser.parse_args()

    rc = RedisClient()
    rc.create_redis_connection(args)
    rc.create_ts((hex(uuid.getnode()) + ':battery'))
    rc.create_ts((hex(uuid.getnode()) + ':power'))

    kws = KeyWordSpotting()
    vad = VAD(rc,kws)

    # Voice User Interface always running in background 
    with sd.InputStream(device=args.device, channels=1, dtype='int16', samplerate=SAMPLING_RATE, blocksize=16000, callback=vad.callback):
        while(True):
            
            key = input()
            if key in ('q', 'Q'):
                print('Stop recording.')
                break


class KeyWordSpotting():
    
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path='./model.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def predict_keyword(self, indata):
        
        my_input = np.array(indata, dtype=np.float32)
        my_input = tf.expand_dims(my_input, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], my_input)
        self.interpreter.invoke()
        my_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return my_output

    def preprocess_audio(self, spectrogram):

        sampling_rate_float32 = tf.cast(SAMPLING_RATE, tf.float32)
        frame_length = int(MEL_LOG_ARGS['frame_length_in_s'] * sampling_rate_float32)
        num_spectrogram_bins = frame_length // 2 + 1

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins= MEL_LOG_ARGS['num_mel_bins'],
            num_spectrogram_bins= num_spectrogram_bins,
            sample_rate= SAMPLING_RATE,
            lower_edge_hertz= MEL_LOG_ARGS['lower_frequency'],
            upper_edge_hertz= MEL_LOG_ARGS['upper_frequency']
        )

        mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)
        log_mel_spectrogram = tf.image.resize(log_mel_spectrogram, [32, 32])

        return log_mel_spectrogram


class VAD():

    def __init__(self, rc, kws):
        # Initial recording status (OFF)
        self.recording_status = False
        # Redis Client
        self.rc = rc
        self.prev_time_s = time()
        self.kws = kws

    def get_audio_from_numpy(self, indata):
        indata = tf.convert_to_tensor(indata, dtype=tf.float32)
        indata = 2*((indata + 32768) / (32767 + 32768))-1
        indata = tf.squeeze(indata)
        return indata

    def get_spectrogram(self, array_audio, downsampling_rate, frame_length_in_s, frame_step_in_s, sampling_rate):
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

    def is_silence(self, array_audio, downsampling_rate, frame_length_in_s, frame_step_in_s, dbFSthres, duration_thres, sampling_rate):

        spectrogram, sampling_rate = self.get_spectrogram(
                array_audio,
                downsampling_rate,
                frame_length_in_s,
                frame_step_in_s,
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

    def callback(self, indata, frames, callback_time, status):

        if(self.recording_status):

            timestamp_ms = int(time() * 1000)
            battery = psu.sensors_battery()

            self.rc.get_client().ts().add(self.rc.get_ts()[0], timestamp_ms, battery.percent)
            self.rc.get_client().ts().add(self.rc.get_ts()[1], timestamp_ms, int(battery.power_plugged))

        array_audio = self.get_audio_from_numpy(indata) 
        silence = self.is_silence(
            array_audio, 
            CALLBACK_ARGS['downsampling_rate'], 
            CALLBACK_ARGS['frame_length_in_s'],
            CALLBACK_ARGS['frame_step_in_s'], 
            CALLBACK_ARGS['dbFSthres'], 
            CALLBACK_ARGS['duration_thres'], 
            SAMPLING_RATE
        )

        if silence == 0:

            zero_padding = tf.zeros(SAMPLING_RATE - tf.shape(array_audio), dtype=tf.float32)
            audio_padded = tf.concat([array_audio, zero_padding], axis=0)
            spectrogram, _ = self.get_spectrogram(audio_padded, CALLBACK_ARGS['downsampling_rate'], CALLBACK_ARGS['frame_length_in_s'], CALLBACK_ARGS['frame_step_in_s'], SAMPLING_RATE)
            input = self.kws.preprocess_audio(spectrogram)
            prediction = tf.squeeze(self.kws.predict_keyword(input))
            print(prediction)
            
            
            prediction_max = prediction[0]
            max_i = 0

            for i in range(len(prediction)):
                if prediction[i] > prediction_max:
                    prediction_max = prediction[i]
                    max_i = i
            

            if(max_i == 1 and prediction_max > 0.95):
                # GO
                if(self.recording_status):
                    print('Recording already in progress!')
                    
                else:
                    self.recording_status = True
                    print(f"Predicted GO with pobability of {prediction_max:.0%}%. Recording Started ...", )

            elif(max_i == 5 and prediction_max > 0.95):
                # STOP
                if(not self.recording_status):
                    print('Recording already stopped!')
                
                else:
                    self.recording_status = False
                    print(f"... Recording Stopped. Predicted STOP with pobability of {prediction_max:.0%}%.")
        

class RedisClient():

    def __init__(self, redis_client = None):
        self.redis_client = redis_client
        self.timeseries_list = []

    def create_redis_connection(self, args):

        print('Creating connection to Redis ...')

        redis_cl = redis.Redis(host=args.host, port=args.port, username=args.user, password=args.password)
        is_connected = redis_cl.ping()

        print('Redis Connected: ', is_connected)

        self.redis_client = redis_cl

    def get_ts(self):
        return self.timeseries_list

    def get_client(self):
        return self.redis_client
    
    def create_ts(self, ts_name, retention_period=0, chnk_size=128):

        if(ts_name not in self.timeseries_list):
            self.timeseries_list.append(ts_name)

        try:
            print('Creating timeseries %s' % (ts_name))

            if(self.redis_client.exists(ts_name) == 0):
                self.redis_client.ts().create(
                    ts_name, 
                    retention_msecs=retention_period,
                    uncompressed=False,
                    chunk_size=chnk_size
                )
                print(f'Timeseries {ts_name} created!')

            else:

                print('Time series %s already exists.' % (ts_name))

                print('Do you want to delete preceding records? [Y/N]')
                    
                while(True):
                    
                    key = input()
                    if key in ('y', 'Y'):
                    
                        timestamp_ms_to = int(time() * 1000)
                        timestamp_ms_from = timestamp_ms_to - int(time() * 1000)
                        self.redis_client.ts().delete(
                            ts_name, 
                            from_time=timestamp_ms_from, 
                            to_time=timestamp_ms_to
                        )
                        print('All preceding data were deleted!')
                        break
                    elif key in ('n', 'N'):
                        break
                    else:
                        print('Sorry, only possible input: [Y/N]')

        except redis.ResponseError:
            print(f'An error occured while creating the timeseries {ts_name}.')
            pass


if __name__ == '__main__':
    main()