import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL, EulerAncestralDiscreteScheduler
import base64
from io import BytesIO

class InferlessPythonModel:
  def initialize(self):
    model_id = "stabilityai/sdxl-turbo"
    snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    self.pipeline = AutoPipelineForText2Image.from_pretrained(model_id,vae=vae, torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
    self.pipeline = self.pipeline.to("cuda")
    self.pipeline.scheduler =  EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

  def infer(self, inputs):
    prompt = inputs["prompt"]
    pipeline_output_image = self.pipeline(prompt=prompt,
                                          num_inference_steps=1,
                                          guidance_scale=1).images[0]
    buff = BytesIO()
    pipeline_output_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return {"generated_image_base64": img_str.decode('utf-8')}

  def finalize(self,args):
    self.pipeline = None
