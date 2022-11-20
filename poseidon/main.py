import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    # Load Model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    # Prompt
    prompt = "a photo of an astronaut riding a horse on mars"

    # Image generation
    image = pipe(prompt).images[0]  
    image.save('result.jpg')

    # Plot
    plt.imshow(image)
    plt.show()