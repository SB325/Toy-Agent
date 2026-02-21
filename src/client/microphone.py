## Connect a microphone and process speech to an audio file
import sounddevice as sd
import numpy as np
import keyboard
import requests
from requests.models import Response
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import io

retry = Retry(
        total=3,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
adapter = HTTPAdapter(max_retries=retry)
session = requests.Session()
session.mount('http://', adapter)

class microphone():
    def __init__(self):
        self.fs = 44100  # Sample rate
        self.recording_list = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        # indata is a numpy array; we copy it to keep it in memory
        self.recording_list.append(indata.copy())

    def record(self):
        print("Recording... Press spacebar to stop.")
        with sd.InputStream(samplerate=self.fs, channels=1, callback=self.callback):
            keyboard.wait('space')

        full_audio = np.concatenate(self.recording_list, axis=0)
        buffer = io.BytesIO()
        np.save(buffer, full_audio)
        self.audio_bytes_with_meta = buffer.getvalue()
        print(f"Stopped. Captured {len(full_audio)/fs:.2f} seconds of audio.")
    
    def send_recording(self, url_in: str):
        # data_in is a lst of numpy objects
        response = self.session.post(
                url=url_in, 
                data=self.audio_bytes_with_meta, 
                headers={'Content-Type': 'application/octet-stream'},
            )
        return response.json()