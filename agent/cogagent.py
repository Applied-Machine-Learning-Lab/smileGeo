# -*- coding: utf-8 -*-
# @Time : 2024/3/15 19:52
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : qwen.py
# @Software: PyCharm

from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import requests

class CogAgent():
    def __init__(self, model="THUDM/cogagent-chat-hf", device=torch.device('cpu'), dtype="bf16", low_cpu_mem=True):
        self.dtype = "bf16"
        if 'public' in model:
            tokenizer_id = 'public/lmsys/vicuna-7b-v1.5'
        else:
            tokenizer_id = 'lmsys/vicuna-7b-v1.5'
        
        self.model_id = model
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id)
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
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.float32).eval().to(self.device)
        elif self.dtype == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(self.device)
        elif self.dtype == 'fp16':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().to(self.device)

    def inference(self, img_url, prompt="", web=False, history=None):
        print('img_url:'. img_url)
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
        else:
            image = Image.open(img_url)

        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])  # chat mode

        if self.dtype == "bf16":
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
                'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
            }
            if 'cross_images' in inputs and inputs['cross_images']:
                inputs['cross_images'] = [[inputs['cross_images'][0].to(self.device).to(torch.bfloat16)]]
        elif self.dtype == "fp16":
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
                'images': [[inputs['images'][0].to(self.device).to(torch.float16)]],
            }
            if 'cross_images' in inputs and inputs['cross_images']:
                inputs['cross_images'] = [[inputs['cross_images'][0].to(self.device).to(torch.float16)]]
        else:
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
                'images': [[inputs['images'][0].to(self.device)]],
            }
            if 'cross_images' in inputs and inputs['cross_images']:
                inputs['cross_images'] = [[inputs['cross_images'][0].to(self.device)]]
        gen_kwargs = {"max_length": 2048, "temperature": 0.9, "do_sample": False}

        response = ''
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
        return response.replace('\n','')