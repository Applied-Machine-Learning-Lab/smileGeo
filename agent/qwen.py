# -*- coding: utf-8 -*-
# @Time : 2024/3/15 19:52
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : qwen.py
# @Software: PyCharm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import requests

class Qwen():
    def __init__(self, model="Qwen/Qwen-VL-Chat", device=torch.device('cpu'), dtype="", low_cpu_mem=True):
        if 'common-crawl-data' in model:
            model_id = "public/Qwen/Qwen-VL-Chat"
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
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
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, fp32=True).eval().to(self.device)
        elif self.dtype == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, bf16=True).eval().to(self.device)
        elif self.dtype == 'fp16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, fp16=True).eval().to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, fp32=True).eval().to(self.device)

    def inference(self, img_url, prompt="", web=False, history=None):
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
            image.save('tmp.jpg', 'JPEG')
            img_url = 'tmp.jpg'
        query = self.tokenizer.from_list_format([{'image': img_url}, {'text': prompt}])
        if history == None:
            response, his = self.model.chat(self.tokenizer, query=query, history=None)
        elif history == 'Yes':
            response, his = self.model.chat(self.tokenizer, query=query, history=self.his)
        self.his = his
        return response.replace('\n','')