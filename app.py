import json
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO


class InferlessPythonModel:
  def initialize(self):
    self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", device_map="auto")

  def infer(self, inputs):
    prompt = inputs["prompt"]
    pipeline_output_image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    buff = BytesIO()
    pipeline_output_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return {"generated_image_base64": img_str.decode('utf-8')}

  def finalize(self,args):
    self.generator = None
    
