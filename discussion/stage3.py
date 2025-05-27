from config import parser
from time import sleep
from utils.my_utils import *
from agent.llava import Llava
from agent.qwen import Qwen
from agent.zephyr import Zephyr
from agent.chatgpt import ChatGPT
from agent.emu2 import Emu2
from agent.claude import Claude
from agent.glm import GLM
import pandas as pd
import math
from utils.logger import Mylogger
from discussion import *

def Stage3(args, model, setting_str, img_url, logger, tmp_city, tmp_country, tmp_confidence, tmp_review_list, tmp_confidence_list):
    prompt = setting_str + 'You say this picture is located in ' + str(tmp_city) + ', ' + str(tmp_country) + ', with confidence: ' + str(tmp_confidence) +'. '
    prompt = prompt + 'Here are others comments with the confidence: '
    for i in range(len(tmp_confidence_list)):
        prompt = prompt + str(tmp_review_list[i]) + ', confidence: ' + str(tmp_confidence_list[i]) + '. '
    prompt = prompt + 'Please summarize all informations by this format: <city>, <country>, <confidence percentage>'
    if not lock_file_check(file_path='tmp/figure/img_lock'):
        sleep(5)
        return
    logger.log('stage 3 prompt: '+prompt)
    logger.log('img url: '+img_url, 'debug')
    if not os.path.exists(img_url):
        lock_remove(file_path='tmp/figure/img_lock')
        return
    answer = model.inference(img_url, prompt=prompt)
    lock_remove(file_path='tmp/figure/img_lock')
    answer_list = answer.split(', ')
    logger.log('stage 3 answer: '+str(answer_list))
    tmp = 2
    while(tmp < len(answer_list)):
        if '%' in answer_list[tmp]:
            tmp_city = answer_list[tmp-2]
            tmp_city = extract_after_substring(tmp_city, 'located in ')
            tmp_city = extract_after_substring(tmp_city, 'located on ')
            tmp_city = extract_after_substring(tmp_city, 'located at ')
            tmp_city = extract_after_substring(tmp_city, 'site in ')
            tmp_country = answer_list[tmp-1]

            ###### tmp confidence ######
            tmp_confidence = answer_list[tmp]
            tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'

            content = str(args.agent_num)+'#'+tmp_city.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'#'+tmp_country.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'#'+tmp_confidence.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'\n'
            add_to_file('tmp/discussion/stage_3_lock', 'tmp/discussion/stage_3.csv', content)
            break
        tmp += 1
    if tmp >= len(answer_list):
        content = str(args.agent_num)+'#None#None#0%\n'
        add_to_file('tmp/discussion/stage_3_lock', 'tmp/discussion/stage_3.csv', content)
    return