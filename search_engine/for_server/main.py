from flask import Flask, request, jsonify
import requests
import asyncio
from typing import Optional
import os
import shutil
from loguru import logger
import itertools
from PicImageSearch import Google, Network
from PicImageSearch.model import GoogleResponse
from PicImageSearch.sync import Google as GoogleSync
from lxml import etree

app = Flask(__name__)

file = "tmp/tmp.jpg"
base_url = "https://www.google.co.jp/searchbyimage"
proxies = None

@logger.catch()
async def test_async():
    async with Network(proxies=proxies) as client:
        google = Google(client=client, base_url=base_url)
        resp = await google.search(file=file)
        print('aaaaaa')
        # show_result(resp, 'a')
        resp2 = await google.next_page(resp)
        print('bbbbbb')
        if resp2:
            # resp3 = await google.pre_page(resp2)
            resp3 = await google.next_page(resp)
            print('cccccc')
            return resp, resp2, resp3
        else:
            return resp, resp2, None

def show_result(resp: Optional[GoogleResponse], char='#') -> None:
    if not resp:
        print(f'{char}: resp error!')
        return
    # logger.info(resp.origin)  # Original Data
    logger.info(resp.pages)
    logger.info(len(resp.pages))
    logger.info(resp.url)  # Link to search results
    logger.info(resp.page_number)

    # try to get first result with thumbnail
    selected = next((i for i in resp.raw if i.thumbnail), resp.raw[0])
    logger.info(selected.origin)
    logger.info(selected.thumbnail)
    logger.info(selected.title)
    logger.info(selected.url)
    logger.info("-" * 50)

@app.route('/imgsearch', methods=['POST'])
def forward_post_request():
    tmp_dir = 'tmp'
    # 遍历目录
    for item_name in os.listdir(tmp_dir):
        # 构建完整的文件或目录路径
        item_path = os.path.join(tmp_dir, item_name)

        try:
            # 检查是文件还是目录
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # 删除文件或链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除目录及其所有内容
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')
    print("'/tmp' directory has been cleared.")

    print('request: ', request)
    # 获取原始POST请求的数据和文件
    data = request.form.to_dict()
    file = request.files

    print('data: ', data)
    print('file: ', file)

    uploaded_file = file.get('file')

    file_path = ""
    if uploaded_file != None:
        print('uploaded_file: ', uploaded_file)

        file_path = 'tmp/tmp.jpg'
        # 保存文件
        uploaded_file.save(file_path)
        print('00000000000')

        resp1, resp2, resp3 = asyncio.run(test_async())
        print('resp1:', resp1)
        print('resp2:', resp2)
        print('resp3:', resp3)
        resp = None
        if resp1 != None:
            resp = resp1
            print('resp1')
        if resp2 != None:
            resp = resp2
            print('resp2')
        if resp3 != None:
            resp = resp3
            print('resp3')
        if resp == None:
        	return {'pages': 0, 'url': 0, 'page_number': 0,'data': []}
        # print(resp)
        # show_result(resp, 'b')
        # 使用生成器表达式和islice来获取前三个满足条件的元素
        selected_items = []
        for i in resp.raw:
            if i.thumbnail:
                selected_items.append(i)
                if len(selected_items) >= 3:
                    break

        data = {'pages': resp.pages, 'url': len(resp.url), 'page_number': resp.page_number,'data': []}
        for item in selected_items:
            data['data'].append({'thumbnail':item.thumbnail,'title':item.title,'url':item.url})

        print('response:', data)

        # 将目标服务器的响应返回给原始请求者
        return data
        
# 定义一个通用的404错误处理器
@app.errorhandler(404)
def page_not_found(e):
    # 打印请求信息
    print_request_info()
    # 对于其他路径，返回404 Not Found
    return jsonify({"error": "Page not found"}), 404

# 定义一个405错误处理器
@app.errorhandler(405)
def method_not_allowed(e):
    # 打印请求信息
    print_request_info()
    # 返回405 Method Not Allowed
    return jsonify({"error": "Method not allowed"}), 405

# 打印请求信息的函数
def print_request_info():
    print("Error handling request:")
    print("Path:", request.path)
    print("Method:", request.method)
    try:
        data = request.get_data(as_text=True)  # 获取文本格式的请求数据
        # 对于Windows CMD，可能需要转换编码
        print("Data:", data.encode('GBK', errors='ignore').decode('GBK'))
    except Exception as e:
        print("Error reading request data:", e)

if __name__ == '__main__':
    app.run(port=5432)