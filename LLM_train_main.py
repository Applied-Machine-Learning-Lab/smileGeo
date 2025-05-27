# -*- coding: utf-8 -*-
# @Time : 2024/3/17 13:35
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : LLM_main.py
# @Software: PyCharm

from agent.agents import Agents
from config import parser
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import GeoDataset
from PIL import Image
import random
from utils.my_utils import *
from utils.geo_utils import *
from utils.logger import Mylogger
from search_engine.img_web_search import google_search_path_only
import pickle as pkl


def train(args):
    lock_remove(file_path='tmp/figure/img_lock')
    # set random seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clear_directory(args.save_path)
    logger = Mylogger('train_main', out_file=args.save_path+'/train_main.log')
    logger.log('Initialize ...')

    adj_next = None

    query_cnt = 0
    data_loader = DataLoader(GeoDataset(root=args.data_root), batch_size=1, shuffle=True)
    # init
    agents = Agents(args, logger, adj=adj_next, load_checkpoint=False)

    
    for ep in range(args.epoch):
        corr_cnt = 0
        total_cnt = 0
        Nagent = agents.get_Nnodes()
        logger.log('Agent number: '+str(Nagent))
        adj_pre = agents.get_adj()
        # logger.log('adj_pre: '+str(adj_pre))
        for i, data in enumerate(data_loader):
            img_name, city, country, label, name, madeby = data
            img_name = img_name[0]
            city = city[0]
            country = country[0]
            Ground_truth = get_location_str(city, country, TOKEN=args.geo_token)
            logger.log(str(Ground_truth), 'debug')
            query_cnt += 1
            logger.log('Epoch: '+str(ep+1)+', Data index: '+str(i)+', query cnt: '+str(query_cnt)+', ground truth: '+city+', '+country, 'debug')
            label = label[0]
            name = name[0]
            madeby = madeby[0]

            img_path = args.data_root + '/pic/' + img_name
            logger.log('img_path: '+str(img_path), 'debug')

            content = []
            tmp_cache_path = args.data_root + '/cache/' + img_name.split('.')[0] + '.pkl'
            if os.path.exists(tmp_cache_path):
                f = open(tmp_cache_path, 'rb')
                content = pkl.load(f)['data']
                f.close()
            # content = google_search_path_only(img_path)
            len_content = len(content)
            web_list = []
            web_list_high_quality = []
            tmp_cnt = 1
            if len_content > 5:
                len_content = 5
            for i in range(len_content):
                thumbnail = content[i]['thumbnail']
                title = content[i]['title']
                sleep(1)
                res = get_location_general(title)
                tmp_ans = title
                if res['has_answer']:
                    tmp_ans = tmp_ans + ', ' + res['city'] + ', ' + res['coutry']
                web_list.append(str(i+1) + ': ' + tmp_ans + '.')
                if 'wiki' in thumbnail or 'flickr' in thumbnail or 'klook' in thumbnail:
                    web_list_high_quality.append(str(tmp_cnt) + ': ' + tmp_ans + '.')
                    tmp_cnt += 1

            image = Image.open(img_path)

            agents.set_image(image)

            select_agent = [0.5 for i in range(Nagent)]
            # logger.log(str(select_agent), 'debug')test

            # multi-round peer reviews
            cnt = 0
            while(cnt < args.max_iter):
                while(True):
                    if not lock_file_check(file_path='tmp/figure/img_lock'):
                        sleep(5)
                        continue
                    break
                logger.log('Reviewing round: '+str(cnt))
                cnt+=1
                # init answer agents
                logger.log('  Init answer agent ...')
                Nanswer_agents = random.randint(1, Nagent)
                answer_agent_list = [] # start from 0
                for i in range(Nanswer_agents):
                    tmp = random.randint(0, Nagent-1)
                    while(True):
                        if tmp not in answer_agent_list:
                            answer_agent_list.append(tmp)
                            break
                        else:
                            tmp = (tmp + 1) % Nagent
                        if len(answer_agent_list) == Nagent:
                                break
                # ######### for test ##########
                # logger.log('    For test only! Fix the answer agent ...', 'debug')
                # answer_agent_list = [0,1,5,2] # start from 0
                # ######### for test ##########
                Nanswer_agents = len(answer_agent_list)
                logger.log('  Answer agent initialized!')

                logger.log('  Init \'tmp\' directory for agent discussion ...')
                str_numbers = [str(number) for number in answer_agent_list]
                tmp_pic_name = "#".join(str_numbers) + '.jpg'
                clear_directory(os.path.join('tmp','figure'))
                clear_directory(os.path.join('tmp','discussion'))
                with open(os.path.join('tmp', 'discussion','stage_1.csv'), 'w') as f:
                    f.write('agentID#city#country#confidence#explain\n')
                with open(os.path.join('tmp', 'discussion','stage_2.csv'), 'w') as f:
                    f.write('agentID#reviewerID#confidence#review\n')
                with open(os.path.join('tmp', 'discussion','stage_3.csv'), 'w') as f:
                    f.write('agentID#final_city#final_country#confidence\n')
                # ######### for test ##########
                # logger.log('    For test only! Fix the files in \'tmp\' ...', 'debug')
                # with open(os.path.join('tmp', 'discussion','sample_stage_1.csv'), 'w') as f:
                #     f.write('agentID#city#country#confidence#explain\n0#beijing#China#None\n1#beijing#China#None\n5#beijing#China#None\n2#beijing#China#None\n')
                # with open(os.path.join('tmp', 'discussion','sample_stage_2.csv'), 'w') as f:
                #     f.write('agentID#reviewerID#confidence#review\n0#4#100%#None\n0#3#100%#None\n1#4#100%#None\n1#3#100%#None\n5#4#100%#None\n5#3#100%#None\n2#4#100%#None\n2#3#100%#None\n')
                # with open(os.path.join('tmp', 'discussion','sample_stage_3.csv'), 'w') as f:
                #     f.write('agentID#final_city#final_country#confidence\n0#beijing#China#100%\n1#beijing#China#100%\n5#beijing#China#100%\n2#beijing#China#100%\n')
                # ######### for test ##########
                image.save(os.path.join('tmp','figure', tmp_pic_name), 'JPEG')
                logger.log('  Ori_img_name: '+img_name+' -> Renamed_img_name: '+tmp_pic_name)

                f = open(os.path.join('tmp','figure', 'web_searching.txt'), 'w')
                if len(web_list_high_quality) >= 3:
                    for line in web_list_high_quality:
                        f.write(line+'\n')
                else:
                    for line in web_list:
                        f.write(line+'\n')
                f.close()


                # init review agents
                logger.log('  Init review agent ...')
                with open(os.path.join('tmp','review_agents.csv'), 'w') as f:
                    f.write("agentID#reviewerID\n")
                Nreview_agents = args.reviewers
                review_agent_list = []  # start from 0
                for answer_agent in answer_agent_list:
                    if args.mode == 'bfs':
                        logger.log('  Using bfs searching of agent '+str(answer_agent) + ' ...', 'debug')
                        neighbor_agents = agents.get_neighbors_with_max_weight_paths(answer_agent, max_hops=args.max_hops)
                        sorted_neighbor_agents = [a[0] for a in dict_sort_by_value(neighbor_agents)]
                    elif args.mode == 'randomwalk':
                        logger.log('  Using random walk searching of agent '+str(answer_agent) + ' ...', 'debug')
                        sorted_neighbor_agents = agents.get_neighbors_with_random_walk(answer_agent, k=args.reviewers+1)
                    else:
                        logger.log('  Mode error! Please check your configures.', 'error')
                        return False
                    if len(sorted_neighbor_agents) > Nreview_agents:
                        review_agent_list = sorted_neighbor_agents[:Nreview_agents]
                    else:
                        review_agent_list = sorted_neighbor_agents
                    for ra in review_agent_list:
                        with open(os.path.join('tmp', 'review_agents.csv'), 'a') as f:
                            f.write(str(answer_agent)+'#'+str(ra)+'\n')
                # ######### for test ##########
                # logger.log('    For test only! Fix the reivew agent ...', 'debug')
                # review_agent_list = [4,3]
                # with open(os.path.join('tmp', 'review_agents.csv'), 'w') as f:
                #     f.write('agentID#reviewerID\n0#0\n6#4\n6#3\n1#4\n1#3\n5#4\n5#3\n2#4\n2#3\n')
                # ######### for test ##########
                Nreview_agents = len(review_agent_list)
                logger.log('  Reviewer agent initialized!')
                lock_remove(file_path='tmp/figure/img_lock')

                # stage 1

                # stage 2

                # stage 3

                logger.log('  Wait until the discussion among agent finished ...')
                total_cnt += 1
                corr_flag = False
                while(True):
                    tmp_path = os.path.join('tmp','discussion', 'stage_3.csv')
                    # ######### for test ##########
                    # logger.log('    For test only! Fix the result file ...', 'debug')
                    # tmp_path = os.path.join('tmp','discussion', 'sample_stage_3.csv')
                    # ######### for test ##########
                    if os.path.exists(tmp_path):
                        stage3_df = pd.read_csv(tmp_path, sep='#')
                        flag = False
                        if len(stage3_df) == Nanswer_agents:
                            flag = True
                        if flag: 
                            for index, row in stage3_df.iterrows():
                                answer_agent = row['agentID']
                                review_agent_df = pd.read_csv(os.path.join('tmp', 'review_agents.csv'), sep='#')
                                reviewers_df = review_agent_df[review_agent_df['agentID'] == answer_agent]
                                tmp_city = row['final_city']
                                tmp_country = row['final_country']
                                tmp_confidence = row['confidence']
                                tmp_confidence = extract_number_before_percent(tmp_confidence)
                                if type(tmp_city) != str:
                                    tmp_city = 'None'
                                if type(tmp_country) != str:
                                    tmp_country = 'None'
                                logger.log('  answer agent: '+str(answer_agent)+'\t'+str(tmp_city)+','+str(tmp_country)+' : '+str(city)+','+str(country), 'debug')
                                Answer = get_location_str(tmp_city, tmp_country, TOKEN=args.geo_token)
                                logger.log('    '+str(Answer), 'debug')
                                query_cnt += 1
                                if are_same_place(Ground_truth, Answer) and tmp_confidence >= args.confidence_threshold:
                                    logger.log('    The answer is correct! Update links ...', 'debug')
                                    if not corr_flag:
                                        corr_flag = True
                                        corr_cnt += 1
                                    for index, row in reviewers_df.iterrows():
                                        agents.enhance_link(answer_agent, row['reviewerID'])
                                        select_agent[int(answer_agent)] = 1.0
                                else:
                                    logger.log('    The answer is wrong! Update links ...', 'debug')
                                    for index, row in reviewers_df.iterrows():
                                        agents.weaken_link(answer_agent, row['reviewerID'])
                                        select_agent[int(answer_agent)] = 0.0
                                logger.log('    Finished!', 'debug')
                            break
                    logger.log('  stage_3.csv is not fully generated! Sleep...', 'debug')
                    sleep(60)
                logger.log('corr_cnt: '+str(corr_cnt)+'\ttotal_cnt: '+str(total_cnt)+'\tpercentage: '+str(corr_cnt/total_cnt*100)+'%', 'info')
                adj_next = agents.get_adj()
                # print('adj_next', adj_next)

            # social network training
            logger.log('Start agent social network training ...')
            with open('graph_tmp_data/data.pkl', 'ab') as f:
                obj = adj_pre, adj_next, select_agent 
                pkl.dump(obj, f)
            agents.social_network_training(adj_pre, adj_next, select_agent)
            logger.log('Finished!')

            logger.log('Set up model checkpoint ...')
            agents.model_save(args.save_path)
            logger.log('Finished!')

            # ######### for test ##########
            # logger.log('For test only! Exit ...', 'debug')
            # break
            # ######### for test ##########
    return

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cpu')
    if args.cuda >= 0:
        args.device = torch.device('cuda:'+str(args.cuda))
    train(args)