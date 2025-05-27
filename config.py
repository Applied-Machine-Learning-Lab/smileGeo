import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'seed': (4567, 'seed for training'),
        'max-iter': (1, 'seed for training'),
        'cuda':(0, 'cuda device'),
        'lr':(0.0001, 'learning rate'),
        'weight-decay':(0.001, 'parameter for optimizer'),
        'stepsize':(100, 'parameter for LR'),
        'gamma': (0.99, 'parameter for LR'),
        'save-path':('saved', 'model save path'),
        'geo-token':(None, 'None means to use osm api [[a5e5b7ab197c4c7b8012d40f3df3ed97]]'),
        'confidence-threshold':(75.0, 'threshold of confidence'),
        'epoch':(1, 'number of epochs')
    },
    'agent_config': {
        'prompt':('', 'some prompts here'),
        'agent-num':(10, 'agent number in agent_settings.csv'),
        'low_cpu_mem':(True, 'use low cpu memory'),
        'dtype':('bf16', 'None string, bf16, fp16'),
        'openai-token':('xxx', 'token for chatgpt'),
        'claude-token':('xxx', 'token for claude'),
        'gemini-token':('xxx', 'token for gemini'),
        'glm-api-key':(None, 'api key for GLM')
    },
    'framework_config': {
        'reviewers':(1, 'number of reviewers'),
        'max-hops':(2, 'max hops in the agent social network'),
        'confidence_th':(75, 'confidence threshold'),
        'vae':('ema', 'type of vae model'),
        'link-threshold':(0.001, 'weight threshold of links'),
        'mode':('randomwalk', 'choice=[randomwalk, bfs]')
    },
    'data_config': {
        'data-root': ('data', 'which dataset to use'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
