from flask import Flask, render_template, request, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)





app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html', )
  
@app.route('/generate', methods=['POST'])
def generate():

  prompt = request.form.get('prompt')
  print("prompt:", prompt)
  image = pipe(prompt).images[0]
  image_path = "static\images\generated_image.png"
  image.save(image_path)

  return render_template('index.html', image_path=image_path)

  
  
  

@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)
