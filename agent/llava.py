# -*- coding: utf-8 -*-
# @Time : 2024/3/15 17:57
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : llava.py
# @Software: PyCharm

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, VipLlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer
from utils.my_utils import extract_after_substring

class Llava():
    def __init__(self, model="llava-hf/llava-1.5-7b-hf", device=torch.device('cpu'), dtype="", low_cpu_mem=True):
        self.model_id = model
        if 'v1.6' in self.model_id:
            if 'common-crawl-data' in self.model_id:
                if '7b' in self.model_id:
                    if 'mistral' in self.model_id:
                        mid = "public/llava-hf/llava-v1.6-mistral-7b-hf"
                    elif 'vicuna' in self.model_id:
                        mid = "public/llava-hf/llava-v1.6-vicuna-7b-hf"
                elif '13b' in self.model_id:
                    mid = 'public/llava-hf/llava-v1.6-vicuna-13b-hf'
                elif '34b' in self.model_id:
                    mid = 'public/llava-hf/llava-v1.6-34b-hf'
                    dtype = 'f16'
                AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            if device == torch.device('cuda'):
                device = torch.device('cuda:0')
                self.devid = int(str(device).split(':')[-1])
            elif device == torch.device('cpu'):
                self.devid = -1
                self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, trust_remote_code=True)
                if '34b' in self.model_id:
                    self.processor = LlavaNextProcessor.from_pretrained(self.model_id, use_fast=False)
                    return
                self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
                return
            else:
                self.devid = int(str(device).split(':')[-1])
            if dtype == 'f16':
                self.dtype =torch.float16
            else:
                self.dtype = torch.float
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=low_cpu_mem,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
        elif 'vip' in self.model_id:
            if 'common-crawl-data' in self.model_id:
                if '7b' in self.model_id:
                    mid = "public/llava-hf/vip-llava-7b-hf"
                elif '13b' in self.model_id:
                    mid = 'public/llava-hf/vip-llava-13b-hf'
                AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            if device == torch.device('cuda'):
                device = torch.device('cuda:0')
                self.devid = int(str(device).split(':')[-1])
            elif device == torch.device('cpu'):
                self.devid = -1
                self.model = VipLlavaForConditionalGeneration.from_pretrained(self.model_id, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                return
            else:
                self.devid = int(str(device).split(':')[-1])
            if dtype == 'f16':
                self.dtype =torch.float16
            else:
                self.dtype = torch.float
            self.model = VipLlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=low_cpu_mem,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
        else:
            if 'common-crawl-data' in self.model_id:
                mid = "public/llava-hf/llava-1.5-7b-hf"
                AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            ###################
            path = self.model_id + "/config.json"  # Replace with the actual path to your config.json file
            # Step 1: Read the file
            with open(path, 'r') as file:
                lines = file.readlines()
            # Step 2: Modify the content
            if len(lines) > 3:
                # Remove the last three lines
                modified_lines = lines[:-3]
            else:
                modified_lines = []
            # Assuming the file is JSON, the last line should properly close the JSON object.
            # We add the new content before the closing brace if it exists, or just append it if not.
            modified_lines.append('"vision_feature_select_strategy": "default"}\n')
            # Step 3: Write back to the file
            with open(path, 'w') as file:
                file.writelines(modified_lines)
            # print(modified_lines)
            ###################
            if device == torch.device('cuda'):
                device = torch.device('cuda:0')
                self.devid = int(str(device).split(':')[-1])
            elif device == torch.device('cpu'):
                self.devid = -1
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id,
                trust_remote_code=True)
                return
            else:
                self.devid = int(str(device).split(':')[-1])
            if dtype == 'f16':
                self.dtype =torch.float16
            else:
                self.dtype = torch.float
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=low_cpu_mem,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
        
        if '34b' in self.model_id:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_id, use_fast=False)
            return
        self.processor = AutoProcessor.from_pretrained(self.model_id)


    def inference(self, img_url, history="", prompt="", web=False):
        if '34b' in self.model_id:
            constant_prompt = '<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n'
        else:
            constant_prompt = "USER: <image>\n"
        if history != None and history != '':
            final_prompt = constant_prompt + history + ' '
        else:
            final_prompt = constant_prompt
        if '34b' in self.model_id:
            final_prompt = final_prompt + prompt + '<|im_end|><|im_start|>assistant\n'
        else:
            final_prompt = final_prompt + prompt + '\nASSISTANT:'
        # print('final_prompt:', final_prompt)
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
        else:
            image = Image.open(img_url)

        # print('image:', image)
        inputs = self.processor(final_prompt, image, return_tensors='pt')
        if self.devid != -1:
            inputs = inputs.to(self.devid, self.dtype)
        # print('inputs:', inputs)
        output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        # print('output:', output)
        res = self.processor.decode(output[0], skip_special_tokens=True)
        # print('res:', res)
        if '34b' in self.model_id:
            res_clean = extract_after_substring(res, 'assistant\n')
        else:
            res_clean = extract_after_substring(res, 'ASSISTANT: ')
        return res_clean.replace('\n','')