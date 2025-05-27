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

from utils.geo_utils import get_location_str, are_same_place
from utils.my_utils import *


def train(args):
    # set random seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adj_next = None

    data_loader = DataLoader(GeoDataset(root=args.data_root), batch_size=1, shuffle=True)

    # init
    agents = Agents(args, adj=adj_next)

    for i, data in enumerate(data_loader):
        img_name, city, country, label, name, madeby = data
        img_name = img_name[0]
        city = city[0]
        country = country[0]
        label = label[0]
        name = name[0]
        madeby = madeby[0]
        Ground_truth = get_location_str(city, country, TOKEN=args.geo_token)

        img_path = args.data_root + '/pic/' + img_name
        image = Image.open(img_path)

        agents.set_image(image)
        Nagent = agents.get_Nnodes()
        adj_pre = agents.get_adj()
        agents.social_network_inference(adj_pre)
        # print('adj_pre:', adj_pre)

        # single-round peer reviews
        # init answer agents
        Nanswer_agents = random.randint(1, Nagent+1)
        answer_agent_list = [] # start from 0
        for i in range(Nanswer_agents):
            tmp = random.randint(0, Nagent)
            while(True):
                if tmp not in answer_agent_list:
                    answer_agent_list.append(tmp)
                    break
                else:
                    if len(answer_agent_list) == Nagent:
                        break
                    tmp = (tmp + 1) % Nagent
        Nanswer_agents = len(answer_agent_list)

        str_numbers = [str(number) for number in answer_agent_list]
        tmp_pic_name = "#".join(str_numbers) + '.jpg'
        clear_directory(os.path.join('tmp','figure'))
        clear_directory(os.path.join('tmp','discussion'))
        with open(os.path.join('tmp', 'discussion','stage_1.csv'), 'w') as f:
            f.write('agentID#city#country#confidence#explain\n')
        with open(os.path.join('tmp', 'discussion','stage_2.csv'), 'w') as f:
            f.write('agentID#reviewerID#confidence#review\n')
        with open(os.path.join('tmp', 'discussion','stage_3.csv'), 'w') as f:
            f.write('agentID#confidence#final_city#final_country#comments\n')
        image.save(os.path.join('tmp','figure', tmp_pic_name), 'JPEG')
        

        # init review agents
        with open(os.path.join('tmp','review_agents.csv'), 'w') as f:
            f.write("agentID#reviewerID\n")
        Nreview_agents = args.reviewers
        review_agent_list = []  # start from 0
        for answer_agent in answer_agent_list:
            if args.mode == 'bfs':
                neighbor_agents = agents.get_neighbors_with_max_weight_paths(answer_agent, max_hops=args.max_hops)
                sorted_neighbor_agents = [a[0] for a in dict_sort_by_value(neighbor_agents)]
            elif args.mode == 'randomwalk':
                sorted_neighbor_agents = agents.get_neighbors_with_random_walk(answer_agent, k=args.reviewers)
            else:
                print('Mode error! Please check your configures.')
                return False
            if len(sorted_neighbor_agents) > Nreview_agents:
                review_agent_list = sorted_neighbor_agents[:Nreview_agents]
            else:
                review_agent_list = sorted_neighbor_agents
            for ra in review_agent_list:
                with open(os.path.join('tmp', 'review_agents.csv'), 'a') as f:
                    f.write(str(answer_agent)+'#'+str(ra)+'\n')
        Nreview_agents = len(review_agent_list)

        # stage 0

        # stage 1

        # stage 2

        while(True):
            tmp_path = os.path.join('tmp','discussion', 'stage_3.csv')
            if os.path.exists(tmp_path):
                stage3_df = pd.read_csv(tmp_path, sep='#')
                flag = False
                if len(stage3_df) == Nanswer_agents:
                    flag = True
                if flag:
                    for index, row in stage3_df.iterrows(): 
                        answer_agent = row['agentID']
                        print('answer agent:', answer_agent)
                        review_agent_df = pd.read_csv(os.path.join('tmp', 'review_agents.csv'), sep='#')
                        reviewers_df = review_agent_df[review_agent_df['agentID'] == answer_agent]
                        tmp_city = row['final_city']
                        tmp_country = row['final_country']
                        tmp_confidence = int(row['confidence'][:-1])
                        tmp_comments = row['comments']
                        print('answer:', tmp_city, tmp_country, tmp_confidence)
                        Answer = get_location_str(tmp_city, tmp_country, TOKEN=args.geo_token)
                        if are_same_place(Answer, Ground_truth) and tmp_confidence > 75:
                            for index, row in reviewers_df.iterrows():
                                agents.enhance_link(answer_agent, row['reviewID'])
                        else:
                            for index, row in reviewers_df.iterrows():
                                agents.weaken_link(answer_agent, row['reviewID'])
                    print('end')
                    break
            print('stage_2.csv is not fully generated! Sleep...')
            sleep(60)
        adj_next = agents.get_adj()


    return

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cpu')
    if args.cuda >= 0:
        args.device = torch.device('cuda:'+str(args.cuda))
    train(args)