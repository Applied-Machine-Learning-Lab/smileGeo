import requests
import os

## please run ''python for_sever/main.py'' on a windows server to use google image search

def google_search(image_path):
    url = 'http://xxx.xxx.xxx.xxx/imgsearch'
    files = {'file': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    files['file'].close

    if response.status_code == 200:
        print('upload success')
        data = response.json()
        # print('data: ', data)
    else:
        print('upload failured!', response.status_code)
        return None

    content = data['data']
    # len_content = len(content)

    # for i in range(len_content):
    #     thumbnail = content[i]['thumbnail']
    #     title = content[i]['title']
    #     url = content[i]['url']
    #     print(title, url)
    return content

def google_search_path_only(image_path):
    url = 'http://xxx.xxx.xxx.xxx/imgsearch'
    response = requests.post(url, data={'img_path': image_path})

    if response.status_code == 200:
        data = response.json()
        # print('data: ', data)
    else:
        print('Failured!', response.status_code)
        return []

    content = data['data']
    return content

if __name__ == '__main__':
    print(google_search_path_only('data/pic/1.jpg'))
