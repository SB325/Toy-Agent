from ..inference import triton_inference

# LLM 
model = "openai/gpt-oss-20b"

client = triton_inference(model)

def image_or_joke(prompt: str):
    full_prompt = f"Here's a request for a joke. Make the joke in accordance with the \
        following prompt: {prompt}"
    result = client.run_inference(full_prompt)
    response = {'returned_value': result}
        
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Joke Generator Model',
                    description='This script accepts a prompt requesting a joke and sends one in respnose.',
                    epilog='by: SFB')
    parser.add_argument('prompt')

    args = parser.parse_args()
    input_prompt = args.prompt
    
    joke = image_or_joke(input_prompt)
    pdb.set_trace()