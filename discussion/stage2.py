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

def Stage2(args, model, setting_str, img_url, tmp_city, tmp_country, tmp_confidence, tmp_explain, logger, ansID):
    prompt = setting_str + 'Someone say this picture is located in ' + str(tmp_city) + ', ' + str(tmp_country) + ', with confidence: ' + str(tmp_confidence) +' (explain: '+str(tmp_explain)+'). What do you think? Please use the following format to answer: <confidence percentage>, <explain>'
    if not lock_file_check(file_path='tmp/figure/img_lock'):
        sleep(5)
        return
    logger.log('img url: '+img_url, 'debug')
    if not os.path.exists(img_url):
        lock_remove(file_path='tmp/figure/img_lock')
        return
    logger.log('stage 2 prompt: '+prompt)
    answer = model.inference(img_url, prompt=prompt)
    lock_remove(file_path='tmp/figure/img_lock')
    answer_list = answer.split(', ')
    logger.log('stage 2 answer: '+str(answer_list))
    tmp = 0
    while(tmp < len(answer_list)):
        if '%' in answer_list[tmp]:
            tmp_confidence = answer_list[tmp]
            tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'
            if tmp+1 >= len(answer_list):
                tmp_explain = substring_after_char(answer_list[tmp], '%')
                if tmp_explain == '' or tmp_explain == None:
                    tmp_explain = 'None'
            else:
                tmp_explain = ', '.join(answer_list[(tmp+1):])
                tmp_explain = remove_trailing_substring(tmp_explain)
            content = str(ansID)+'#'+str(args.agent_num)+'#'+str(tmp_confidence)+'#'+tmp_explain.replace('\"', ' ').replace('\'', ' ').replace('#',' ')+'\n'
            add_to_file('tmp/discussion/stage_2_lock', 'tmp/discussion/stage_2.csv', content)
            break
        tmp += 1
    if tmp >= len(answer_list):
        content = str(ansID)+'#'+str(args.agent_num)+'#0%#None\n'
        add_to_file('tmp/discussion/stage_2_lock', 'tmp/discussion/stage_2.csv', content)
    return