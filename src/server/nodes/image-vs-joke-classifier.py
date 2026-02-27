from ..inference import triton_inference
import pdb

# LLM 
model = "openai/gpt-oss-20b"

client = triton_inference(model)

def image_or_joke(prompt: str):
    full_prompt = f"Determine whether the following text is a request for for an image or a joke. \
            If an image, reply 'image', if a joke, reply 'joke'. Else reply 'fail': {prompt}"
    result = client.run_inference(full_prompt)

    if 'image' in result:
        response = {'result': 'image', 'returned_value': result}
    elif 'joke' in result:
        response = {'result': 'joke', 'returned_value': result}
    else:
        response = {'result': 'fail', 'returned_value': result}

    return response