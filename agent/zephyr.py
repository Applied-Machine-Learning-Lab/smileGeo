# -*- coding: utf-8 -*-
# @Time : 2024/3/15 20:26
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : zephyr.py
# @Software: PyCharm

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import requests

class Zephyr():
    def __init__(self, model="Infi-MM/infimm-zephyr", device=torch.device('cpu'), dtype="", low_cpu_mem=True):
        if 'common-crawl-data' in model:
            model_id = "public/Infi-MM/infimm-zephyr"
            AutoTokenizer.from_pretrained("public/Infi-MM/infimm-zephyr")

        self.model_id = model
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        self.device = device
        self.dtype = dtype
        self.his = None

        if self.device == torch.device('cuda'):
            self.device = torch.device('cuda:0')
            self.devid = int(str(device).split(':')[-1])
        elif self.device == torch.device('cpu'):
            self.devid = -1
        else:
            self.devid = int(str(device).split(':')[-1])

        if self.device == torch.device('cpu'):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, local_files_only=True,trust_remote_code=True).eval()
        elif self.dtype == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,local_files_only=True,torch_dtype=torch.bfloat16,trust_remote_code=True).eval().to(self.device)
        elif self.dtype == 'fp16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,trust_remote_code=True,torch_dtype=torch.float16).eval().to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,trust_remote_code=True).eval().to(self.device)

    def inference(self, img_url, prompt="", web=False, history=None):
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
            image.save('tmp.jpg', 'JPEG')
            img_url = 'tmp.jpg'
        input_list = [{"role": "user","content": [{"image": img_url},prompt]}]
        inputs = self.processor(input_list)
        if self.device != torch.device('cpu'):
            inputs = inputs.to(self.model.device)
        if self.dtype == 'bf16':
            inputs["batch_images"] = inputs["batch_images"].to(torch.bfloat16)
        elif self.dtype == 'fp16':
            inputs["batch_images"] = inputs["batch_images"].to(torch.float16)
        generated_ids = self.model.generate(**inputs,min_generation_length=0,max_generation_length=256)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[-1].replace('\n','')
