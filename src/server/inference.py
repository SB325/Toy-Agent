import numpy as np
import tritonclient.http as httpclient
import argparse
import subprocess
from PIL import Image

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

class triton_inference():
    def __init__(self, model: str):
        self.model_name = model

        # 1. Initialize the client
        # Use "localhost:8001" and tritonclient.grpc if using gRPC
        self.client = httpclient.InferenceServerClient(url=f"{get_triton_ip()}:8000")

        self.outputs = [
            httpclient.InferRequestedOutput("OUTPUT0")
        ]

    def run_inference(self, input_data, to_file: bool = False, verbose: bool = False)
        prompt = np.load(io.BytesIO(numpy_audio_file.data))
        if isinstance(input_data, str):
            prompt = np.array([input_data.encode('utf-8')], dtype=np.object_)

        # 3. Setup Input and Output objects 
        self.inputs = [
            # httpclient.InferInput("prompt", [1], "BYTES"),
            httpclient.InferInput("INPUT0", prompt, "FP32")
        ]
        inputs[0].set_data_from_numpy(prompt)

        # 4. Perform Inference
        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=self.outputs)

        # 5. Get results as a NumPy array
        result = response.as_numpy("OUTPUT0")
        
        if to_file:
            img = Image.fromarray(result.astype(np.uint8))
            img.save("resulting_image.png")
            print("Image saved as resulting_image.png")
            return False
        else:
            if verbose:
                print(result)
        
            return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Triton Client Model Inference',
                    description='This script contains a class that abstracts the Triton \
                        model inference step for agents.',
                    epilog='by: SFB')
    parser.add_argument('input')

    args = parser.parse_args()
    input_data = args.input
    
    client = triton_inference()

    async def transcribe_audio(request: Request):
    input_data = "Hello there."
    result = client.run_inference(input_data)
    
    return {"text": result["text"]}