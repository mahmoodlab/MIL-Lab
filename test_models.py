from transformers import AutoConfig, AutoModel
from src.models import abmil
from src.builder import create_model, save_model
from pathlib import Path
import argparse
import torch
import torch.nn as nn

import pdb

# model_names = ['abmil', 'transmil', 'transformer', 'dftd', 'ilra', 'rrt', 'wikg', 'dsmil', 'clam']
sample_data = torch.randn(1, 5000, 1024)
model_names = ['abmil', 'transmil', 'transformer', 'dftd', 'ilra', 'rrt', 'wikg', 'dsmil', 'clam']
# model_names = ['abmil']
variants = ['base']
encoders = ['conch_v15']
# tasks = ['pc108-24k'] # , 'pc43']
tasks = ['none'] # , 'pc43']
sample_data = torch.randn(1, 5000, 768)
loss_fn = nn.CrossEntropyLoss()
label = torch.LongTensor([0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
repo_name = 'MahmoodLab'
for model_name in model_names:
    for variant in variants:
        for encoder in encoders:
            for task in tasks:
                print(f"Testing {model_name}.{variant}.{encoder}.{task}...")
                config_name = f"{model_name}.{variant}.{encoder}.{task}"
                #config = AutoConfig.from_pretrained(config_name)
                #model = AutoModel.from_pretrained(config_name)
                # model = create_model('abmil')
                # model = create_model('abmil.base')
                # model = create_model('abmil.base.uni')
                # print(f'load from pretrained model')
                # model = create_model(config_name, num_classes=0) #, from_pretrained=True)
                # then save model
                # save_model(model, config_name, save_pretrained=True)

                # print(f'create pretrained model')
                model = create_model(config_name, num_classes=num_classes)
                print(f'load random model')
                print(f"Model {model}.{variant}.{encoder}.{task} tested successfully.")
                model.eval()
                model.to(device)
                
                sample_data = sample_data.to(device)
                label = label.to(device)

                results_dict, log_dict = model.forward(sample_data, label=label, loss_fn=loss_fn,
                                                       return_attention=True)
                assert 'logits' in results_dict and results_dict['logits'].shape[-1] == num_classes
                assert 'loss' in results_dict
                print(f"Model {model}.{variant}.{encoder}.{task} tested successfully.")
                print(f'try saving model')
                save_model(model, config_name)
                print(f'reload model from pretrained')
                create_model(config_name, from_pretrained=True)
                print("-"*100)
