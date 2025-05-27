# -*- coding: utf-8 -*-
# @Time : 2024/3/20 20:26
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : zephyr.py
# @Software: PyCharm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from PIL import Image
import requests

class Emu2():
    def __init__(self, model="BAAI/Emu2-Chat", device=torch.device('cpu'), dtype="", low_cpu_mem=True):
        self.model_id = model
        self.device = device
        self.dtype = dtype
        self.his = None

        if self.device == torch.device('cuda'):
            self.device = torch.device('cuda:0')
            self.devid = int(device.split(':')[-1])
        elif self.device == torch.device('cpu'):
            self.devid = -1
        else:
            self.devid = int(str(device).split(':')[-1])
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_4bit=True, trust_remote_code=True, bnb_4bit_compute_dtype=torch.float16).eval()

    def inference(self, img_url, prompt="", web=False, history=None):
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
            image.save('tmp.jpg', 'JPEG')
            img_url = 'tmp.jpg'
        image = Image.open(img_url).convert('RGB')
        
        query = '[<IMG_PLH>]' + prompt
        inputs = self.model.build_input_ids(text=[query], tokenizer=self.tokenizer, image=[image])

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.float16), # should be torch.float16
                max_new_tokens=64,
                length_penalty=50)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text.replace('\n','')
