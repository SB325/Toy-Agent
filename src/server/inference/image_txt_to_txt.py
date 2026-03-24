import os, sys
import pdb
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer

load_dotenv()
OCR_DIR = os.getenv("OCR_MODEL_STORAGE")

class smart_ocr():
    def __init__(self):
        # self.model_name = model_name
        # It is recommended to use the tokenizer from the base (non-GGUF) model 
        # for stability and performance.
        self.llm = LLM(
            model=f"{OCR_DIR}",
            tokenizer=f"{OCR_DIR}",
            hf_config_path=f"{OCR_DIR}",
            gpu_memory_utilization=0.8, 
            # enforce_eager=True    # Use if memory is tight
            max_model_len=4096,
            max_cudagraph_capture_size=4,
        )
        # Resolution            |	Calculation	Estimated Tokens
        # 512 × 512	            |  256 tokens
        # 1024 × 1024           |  1,024 tokens
        # Full HD (1920 × 1080) |  2,040 tokens

    def decode_image(self, image_file: str):
        extension = os.path.splitext(image_file)[1]
        standard_img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

        if 'pdf' in extension:
            # This returns a list of PIL Image objects (one for each page)
            pages = convert_from_path(image_file, dpi=300)
            # Take the first page to use with your vLLM prompt
            self.image = pages[0].convert("RGB")
        elif extension in standard_img_extensions:
            self.image = Image.open(image_file).convert("RGB")
        else:
            print(f"Extension {extension} is not recognized.")

    def inference(self, 
            image: str,
            prompt_str: str = "Describe this image in detail.",
            verbose: bool = False):
        # Prepare a multimodal prompt (text + image)
        self.decode_image(image)

        tokenizer = AutoTokenizer.from_pretrained(OCR_DIR, trust_remote_code=True)
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_str},
                    ],
                }
            ]

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": self.image}
        }
        # Batching: If you want to process multiple images in a single prompt 
        # (if supported by your specific vLLM version), you would pass a list 
        # of these objects: {"image": [image1, image2]}.
        sampling_params = SamplingParams(temperature=0.2, max_tokens=512)
        outputs = self.llm.generate([inputs], sampling_params=sampling_params)

        for output in outputs:
            for completion in output.outputs:
                reason = completion.finish_reason
                if reason == "length":
                    print("⚠️ The response was cut off because it hit the 'max_tokens' limit.")
                elif reason == "stop":
                    print("✅ The model finished its response naturally.")
                elif reason == "abort":
                    print("❌ The request was terminated prematurely.")

        if verbose:
            # Print the result
            for output in outputs:
                print(f"Generated text: {output.outputs[0].text}")

        return outputs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: image_txt_to_txt.py <path to pdf or image file.>")
        sys.exit(1)
    
    image_file = sys.argv[1] 

    ocrAI = smart_ocr()

    outputs = ocrAI.inference(image = image_file)
    pdb.set_trace()