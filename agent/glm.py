import requests
import json
import base64
from PIL import Image
import io
import time
import jwt

def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


class GLM():
    def __init__(self, token=None):
        self.token = token
        if token == None or token == '':
            raise ValueError("GLM API KEY Error!!!")

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
        "Authorization": f"Bearer {generate_token(API_KEY, 10)}"
        }

        data = {
            "model": "glm-4v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": "Where is this image located?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url" : f"{base64_image}"
                        }
                    }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json=data)

        data = response.json()

        return data['choices']['message']['content'].replace('\n','')