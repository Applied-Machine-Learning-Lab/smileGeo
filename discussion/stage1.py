from config import parser
from time import sleep
from utils.my_utils import *
import pandas as pd
import math
from utils.logger import Mylogger
from discussion import *
import os
from search_engine.img_web_search import google_search
from utils.geo_utils import get_location_general
import re

def Stage1(args, model, setting_str, img_url, logger, prompt_template=1):
    if prompt_template == 1:
        prompt = setting_str + 'Answer the location of this picture by this format: <city>, <country>, <confidence percentage>, <explain>. For example, Paris, France, 60%, *.'
    else:
        prompt = setting_str + 'Answer the location of this picture by this format: <city>, <country>, <confidence percentage>, <explain>. For example, Paris, France, 60%, *.'
    # # sleep(5)
    # content = google_search(img_url)
    # if len(content) != 0:
    #     prompt = prompt + ' If you need to more information about this picture, I have done an internet search for you and the top results are listed below:'
    #     len_content = len(content)
    #     for i in range(len_content):
    #         thumbnail = content[i]['thumbnail']
    #         title = content[i]['title']
    #         sleep(1)
    #         res = get_location_general(title)
    #         tmp_ans = title
    #         if res['has_answer']:
    #             tmp_ans = tmp_ans + ', ' + res['city'] + ', ' + res['coutry']
    #         prompt = prompt + ' ' + str(i+1) + ': ' + tmp_ans + '.'
    #         url = content[i]['url']
    web_search_path = os.path.join('tmp', 'figure', 'web_searching.txt')
    lines = []
    if os.path.exists(web_search_path):
        with open(web_search_path, 'r') as f:
            lines = f.readlines()
    if len(lines) > 0:
        prompt = prompt + '\nIf you need to more information about this picture, I have done an internet search for you and the top results are listed below:'
        cnt = 0
        for line in lines:
            if cnt == 3:
                break
            cnt += 1
            prompt = prompt + ' ' + line[:-1]

    if not lock_file_check(file_path='tmp/figure/img_lock'):
        sleep(5)
        return
    logger.log('stage 1 prompt: '+prompt)
    logger.log('img url: '+img_url, 'debug')
    if not os.path.exists(img_url):
        lock_remove(file_path='tmp/figure/img_lock')
        return
    answer = model.inference(img_url, prompt=prompt)
    lock_remove(file_path='tmp/figure/img_lock')
    answer_list = re.split(r'[.,()]', answer) #answer.split(', ')
    answer_list = [item.strip() for item in answer_list if item.strip()]
    logger.log('stage 1 answer: '+str(answer_list))
    tmp = 1
    while(tmp < len(answer_list)):
        if '%' in answer_list[tmp]:

            ###### tmp city ######
            if tmp-1 ==0:
                tmp_city = 'None'
            elif tmp-1 > 0:
                tmp_city = ', '.join(answer_list[:(tmp-1)])
            else:
                continue
            tmp_city = extract_after_substring(tmp_city, 'located in ')
            tmp_city = extract_after_substring(tmp_city, 'located on ')
            tmp_city = extract_after_substring(tmp_city, 'located at ')
            tmp_city = extract_after_substring(tmp_city, 'site in ')

            ###### tmp country ######
            tmp_country = answer_list[tmp-1]


            ###### tmp confidence ######
            tmp_confidence = answer_list[tmp]
            tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'

            ###### tmp explain ######
            if tmp+1 >= len(answer_list):
                tmp_explain = substring_after_char(answer_list[tmp], '%')
                if tmp_explain == '' or tmp_explain == None:
                    tmp_explain = 'None'
            else:
                tmp_explain = ', '.join(answer_list[(tmp+1):])
                tmp_explain = remove_trailing_substring(tmp_explain, ')')
                tmp_explain = remove_trailing_substring(tmp_explain, '.')
                tmp_explain = remove_trailing_substring(tmp_explain, ')')
            content = str(args.agent_num)+'#'+tmp_city.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'#'+tmp_country.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'#'+tmp_confidence.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'#'+tmp_explain.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'\n'
            add_to_file('tmp/discussion/stage_1_lock', 'tmp/discussion/stage_1.csv', content)
            break
        tmp += 1
    if tmp >= len(answer_list):
        content = str(args.agent_num)+'#None#None#0%#None\n'
        add_to_file('tmp/discussion/stage_1_lock', 'tmp/discussion/stage_1.csv', content)
    return