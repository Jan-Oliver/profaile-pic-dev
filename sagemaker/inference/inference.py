import base64
import torch
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionPipeline

# Based on https://github.com/huggingface/notebooks/blob/main/sagemaker/23_stable_diffusion_inference/sagemaker-notebook.ipynb

from diffusers import StableDiffusionDepth2ImgPipeline, DDIMScheduler

def model_fn(model_dir):
    # Load stable diffusion depth2img and move it to the GPU
    # Use DDIM scheduler
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

# helper decoder
def decode_base64_image(image_string):
  base64_image = base64.b64decode(image_string)
  buffer = BytesIO(base64_image)
  return Image.open(buffer)

def predict_fn(data, pipe):
    # get prompt & parameters
    pos_prompt = data.pop("pos_prompt", data)
    neg_prompt = data.pop("neg_prompt", data)
    init_image_base64 = data.pop("depth_image_base64", data)
    init_image = decode_base64_image(init_image_base64)
    strength = data.pop("strength", 0.825)
    num_images_per_prompt = data.pop("num_images_per_prompt", 10)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    SEED = data.pop("seed", 52362)
    g_cuda = torch.Generator(device='cuda').manual_seed(SEED)

    # run generation with parameters
    generated_images = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}