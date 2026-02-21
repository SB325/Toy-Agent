import torch
from transformers import pipeline
import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI, Request

load_dotenv(override=True)
in_docker = os.getenv("INDOCKER")

def get_triton_ip():
    if bool(in_docker):
        return 'triton'
    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        "docker inspect --format='{{.NetworkSettings.Networks.homeserver.IPAddress}}' triton",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')
    return ip

class speech():
    def __init__(self, model="openai/whisper-small"):
        self.model_name = model

        # 1. Initialize the client
        # Use "localhost:8001" and tritonclient.grpc if using gRPC
        self.client = httpclient.InferenceServerClient(url=f"{get_triton_ip()}:8000")

        self.outputs = [
            httpclient.InferRequestedOutput("OUTPUT0")
        ]

    def transcribe(self, numpy_audio_file, verbose: bool = False)
        audio_data = np.load(io.BytesIO(numpy_audio_file.data))
        # 2. Prepare your input data (match names/shapes in your config.pbtxt)
        # input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_data = audio_data

        # 3. Setup Input and Output objects
        self.inputs = [
            httpclient.InferInput("INPUT0", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)

        # 4. Perform Inference
        response = self.client.infer(self.model_name, inputs, outputs=self.outputs)

        # 5. Get results as a NumPy array
        result = response.as_numpy("OUTPUT0")
        
        if verbose:
            print(result)
        
        return result

### API ###
app = FastAPI(root_path="/transcriber/api/v1")
speech_client = speech()

@app.post("/transcribe")
async def transcribe_audio(request: Request):
    # 1. Read the raw binary data from the request body
    body = await request.body()
    
    # 2. Convert bytes back to a NumPy array
    # IMPORTANT: Use the same dtype (e.g., float32) as your client-side recording
    result = speech_client.transcribe(body)

    # 3. (Optional) Reshape if your recording was multi-channel
    # audio_data = audio_data.reshape(-1, 1) 
    
    return {"text": result["text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)