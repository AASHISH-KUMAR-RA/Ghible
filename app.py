"""
GhibliArt - Transform your photos into Studio Ghibli-style artwork

This script uses the Stable Diffusion model fine-tuned on Studio Ghibli artwork
to create magical, anime-style transformations of input images.
"""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """
    Load the Ghibli-style Stable Diffusion model.
    
    Returns:
        StableDiffusionImg2ImgPipeline: The loaded model pipeline
    """
    model_id = "nitrosocke/Ghibli-Diffusion"
    try:
        logger.info("Loading model...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None  # Disable safety checker for faster processing
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        if device == "cpu":
            pipe.enable_sequential_cpu_offload()
        logger.info(f"Model loaded successfully on {device}!")
        return pipe
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_ghibli_image(image, pipe, strength=0.75):
    """
    Generate a Ghibli-style version of the input image.
    
    Args:
        image (PIL.Image): Input image to transform
        pipe (StableDiffusionImg2ImgPipeline): The model pipeline
        strength (float): How much to transform the image (0.0 to 1.0)
        
    Returns:
        PIL.Image: The transformed Ghibli-style image
    """
    try:
        image = image.convert("RGB")
        width, height = image.size
        
        # Calculate new dimensions to maintain aspect ratio but fit within 512x512
        ratio = min(512/width, 512/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        image = image.resize((new_width, new_height))
        
        prompt = "studio ghibli style, anime, beautiful, detailed, vibrant colors, soft lighting, magical atmosphere, high quality"
        negative_prompt = "ugly, blurry, low quality, distorted, deformed, bad anatomy, disfigured"
        
        logger.info("Generating Ghibli-style image...")
        start_time = time.time()
        
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images[0]
            
        logger.info(f"Image generated in {time.time() - start_time:.2f} seconds!")
        return result
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise

def process_images(original_path, save_path=None):
    """
    Process an input image and generate its Ghibli-style version.
    
    Args:
        original_path (str): Path to the input image
        save_path (str, optional): Path to save the output image
    """
    try:
        # Load original image
        original = Image.open(original_path)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display original image
        ax1.imshow(original)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Load model and generate Ghibli-style image
        pipe = load_model()
        result = generate_ghibli_image(original, pipe)
        
        # Display Ghibli-style image
        ax2.imshow(result)
        ax2.set_title("Ghibli Style")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the output image
        if save_path:
            output_path = save_path
        else:
            output_path = f"ghibli_style_{original_path.split('/')[-1]}"
        result.save(output_path)
        logger.info(f"Image saved as {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert an image to Ghibli style')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, help='Path for the output image (optional)', default=None)
    args = parser.parse_args()
    
    process_images(args.image_path, args.output)