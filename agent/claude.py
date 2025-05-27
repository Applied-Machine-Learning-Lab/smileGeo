import requests
import json
import base64
from PIL import Image
import io

class Claude():
    def __init__(self, token=None):
        self.token = token
        if token == None or token == '':
            raise ValueError("Claude Token Error!!!")

    def inference(self, img_url, prompt="", web=False, history=None):
        if web:
            image = Image.open(requests.get(img_url, stream=True).raw)
            image.save('tmp.jpg', 'JPEG')
            img_url = 'tmp.jpg'

        img_file = open(img_url, 'rb')
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
        img_file.close()

        headers = {
            "x-api-key": f"{self.token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        message = {
            "model":"claude-3-opus-20240229",
            "max_tokens":1024,
            "messages":[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/jpeg',
                                "data": img_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ]
        }
        response = requests.post("https://api.claude-plus.top/v1/messages", headers=headers, json=message)
        # response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)

        print(response.text)

        data = response.json()

        print('answer test:', data)

        return data['content']['text'].replace('\n','')