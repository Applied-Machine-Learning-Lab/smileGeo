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
from agent.cogvlm import CogVLM
from agent.cogagent import CogAgent
import pandas as pd
import math
from utils.logger import Mylogger
from discussion import stage1
from discussion import stage2
from discussion import stage3

def start_agent(args):
    logger = Mylogger('agent_'+str(args.agent_num), out_file=args.save_path+'/agent_'+str(args.agent_num)+'.log')

    agent_df = pd.read_csv('agent/agent_settings.csv', sep='#')

    filtered_df = agent_df[agent_df['index'] == args.agent_num]

    model_name = filtered_df['name'].tolist()[0]
    setting_str = filtered_df['setting'].tolist()[0]
    if setting_str == None or math.isnan(setting_str):
        setting_str = ''
    else:
        setting_str = setting_str + ' '

    model = None
    # load model parameters
    if 'llava' in model_name.lower():
        model = Llava(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)
    elif 'qwen' in model_name.lower():
        model = Qwen(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)
    elif 'zephyr' in model_name.lower():
        model = Zephyr(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)
    elif 'emu2' in model_name.lower():
        model = Emu2(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)
    elif 'chatgpt' in model_name.lower():
        model = ChatGPT(token=args.openai_token)
    elif 'claude' in model_name.lower():
        model = Claude(token=args.claude_token)
    elif 'glm' in model_name.lower():
        model = GLM(token=args.glm_api_key)
    elif 'cogvlm' in model_name.lower():
        model = CogVLM(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)
    elif 'cogagent' in model_name.lower():
        model = CogAgent(model=model_name, device=args.device, dtype=args.dtype, low_cpu_mem=args.low_cpu_mem)

    # total three stages for discussion
    while(True):
        pic_name = ''
        # stage 1 checking
        jpg_names = find_jpg_files('tmp/figure')
        if len(jpg_names) > 0 and pic_name != jpg_names[0]:
            img_url = 'tmp/figure/' + jpg_names[0]
            agent_list = [int(a) for a in jpg_names[0].split('.')[0].split('#')]

            stage_1_df = pd.read_csv('tmp/discussion/stage_1.csv', sep='#')
            if args.agent_num in agent_list and stage_1_df[stage_1_df['agentID'] == args.agent_num].empty:
                stage1.Stage1(args, model, setting_str, img_url, logger)

            # stage 2 checking
            stage_2_agent_df = pd.read_csv('tmp/review_agents.csv', sep='#')
            tmp_s2_agent_df = stage_2_agent_df[stage_2_agent_df['reviewerID']==args.agent_num]
            if not tmp_s2_agent_df.empty:
                stage_1_df = pd.read_csv('tmp/discussion/stage_1.csv', sep='#')

                # ans_agent in review_agents.csv
                s1_agents = []
                for index, row in tmp_s2_agent_df.iterrows():
                    s1_agentID = row['agentID']
                    s1_agents.append(s1_agentID)
                logger.log('s1_agents: '+str(s1_agents), 'debug')
                    
                s2_df = pd.read_csv('tmp/discussion/stage_2.csv', sep='#')
                flag = False
                for ansID in s1_agents:
                    # did not generate s2 review before
                    if s2_df[(s2_df['agentID']==ansID) & (s2_df['reviewerID']==args.agent_num)].empty and (not stage_1_df[stage_1_df['agentID'] == ansID].empty):
                        tmp_s1_df = stage_1_df[stage_1_df['agentID'] == ansID]
                        tmp_city = None
                        tmp_country = None
                        tmp_confidence = None
                        tmp_explain = None
                        for index, row in tmp_s1_df.iterrows():
                            tmp_city = row['city']
                            tmp_country = row['country']
                            tmp_confidence = row['confidence']
                            tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'
                            tmp_explain = row['explain']
                        stage2.Stage2(args, model, setting_str, img_url, tmp_city, tmp_country, tmp_confidence, tmp_explain, logger, ansID)

            # stage 3 checking
            s2_df = pd.read_csv('tmp/discussion/stage_2.csv', sep='#')
            tmp_s2_df = s2_df[s2_df['agentID'] == args.agent_num]
            tmp_s2_agent_df = stage_2_agent_df[stage_2_agent_df['agentID']==args.agent_num]
            if not tmp_s2_df.empty and len(tmp_s2_df) == len(tmp_s2_agent_df):
                s3_df = pd.read_csv('tmp/discussion/stage_3.csv', sep='#')
                if s3_df[s3_df['agentID']==args.agent_num].empty:
                    tmp_confidence_list = []
                    tmp_review_list = []
                    for index, row in tmp_s2_df.iterrows():
                        tmp_confidence = row['confidence']
                        tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'
                        tmp_confidence_list.append(tmp_confidence)
                        tmp_review = row['review']
                        tmp_review_list.append(tmp_review)
                    #
                    s1_df = pd.read_csv('tmp/discussion/stage_1.csv', sep='#')
                    tmp_s1_df = s1_df[s1_df['agentID'] == args.agent_num]
                    tmp_city = None
                    tmp_country = None
                    tmp_confidence = None
                    tmp_explain = None
                    for index, row in tmp_s1_df.iterrows():
                        tmp_city = row['city']
                        tmp_country = row['country']
                        tmp_confidence = row['confidence']
                        tmp_confidence = str(extract_number_before_percent(tmp_confidence))+'%'
                        tmp_explain = row['explain']
                    #
                    stage3.Stage3(args, model, setting_str, img_url, logger, tmp_city, tmp_country, tmp_confidence, tmp_review_list, tmp_confidence_list)
            logger.log('Sleep ...')
        sleep(30)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cpu')
    if args.cuda >= 0:
        args.device = torch.device('cuda:'+str(args.cuda))
    start_agent(args)