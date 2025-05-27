import requests
import json
import base64
from PIL import Image
import io

class ChatGPT():
    def __init__(self, token=None, model="gpt-4-vision-preview"):
        self.token = token
        self.model = model
        if token == None or token == '':
            raise ValueError("ChatGPT Token Error!!!")

    def inference(self, img_url, prompt="", web=False, history=None):
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
            image.save('tmp.jpg', 'JPEG')
            img_url = 'tmp.jpg'

        img_file = open(img_url, 'rb')
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
        img_file.close()

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
        "model": self.model,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_data}",
                    "detail": "low"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
        # please revise the url to one-api you deployed
        response = requests.post("https://master/v1/chat/completions", headers=headers, json=payload)

        data = response.json()

        print(data)

        return data['choices'][0]['message']['content'].replace('\n','')