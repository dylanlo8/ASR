from typing import List
import numpy as np
import logging
import grpc
from tempfile import NamedTemporaryFile
import soundfile as sf
from proto import vad_pb2, vad_pb2_grpc

HOST = "localhost"
PORT = 60053

class VADIterator:
    def __init__(self,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.elapsed_samples = 0
        
        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.elapsed_samples = 0

    def __call__(self, x: np.ndarray, return_seconds=False):
        """
        x: np.ndarray
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """
        with grpc.insecure_channel(f"{HOST}:{PORT}") as channel:
            window_size_samples = len(x[0]) if np.ndim(x) == 2 else len(x)
            self.current_sample += window_size_samples
            self.elapsed_samples += window_size_samples
            
            stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
            request = vad_pb2.VoiceActivityDetectorRequest(audio_data=stream_to_bytes(x))
            response = stub.detect_voice_activity(request)

            speech_prob = response.confidence
            
            logging.info("Voice Detected Confidence: %0.2f", speech_prob)
            t = self.elapsed_samples / self.sampling_rate
            threshold = exponential_threshold(t, self.threshold, 1, 25, 2)
            
            if (speech_prob >= threshold) and self.temp_end:
                self.temp_end = 0

            if (speech_prob >= threshold) and not self.triggered:
                self.triggered = True
                self.elapsed_samples = 0
                speech_start = self.current_sample - self.speech_pad_samples - window_size_samples
                return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

            if (speech_prob < threshold - 0.15) and self.triggered:
                if not self.temp_end:
                    self.temp_end = self.current_sample
                if self.current_sample - self.temp_end < self.min_silence_samples:
                    return None
                else:
                    speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                    self.temp_end = 0
                    self.triggered = False
                    return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}
            return None

def exponential_threshold(
    t: float, 
    start_value: float, 
    end_value: float, 
    duration: float, 
    k: float
) -> float:
    """
    Computes the threshold value using an exponential function.
    Parameters:
    t (float): The elapsed time.
    start_value (float): The initial threshold value.
    end_value (float): The final threshold value.
    duration (float): The duration over which the threshold increases.
    k (float): A constant that controls the rate of increase.

    Returns:
    float: The threshold value at time t.
    """
    return start_value + (end_value - start_value) * (np.exp(k * t / duration) - 1) / (np.exp(k) - 1)

def stream_to_bytes(audio_stream: np.ndarray, sampling_rate: int = 16000) -> bytes:
    
    with NamedTemporaryFile(suffix=".wav") as f:
        sf.write(
            file=f, 
            data=audio_stream,
            samplerate=sampling_rate, 
            format='WAV', 
            subtype='PCM_16'
        )
        f.seek(0)
        return f.read()