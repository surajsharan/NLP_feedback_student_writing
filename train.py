import config
import engine
import dataset
from model import LongformerModel
import preprocessing
import pandas as pd
import numpy as np
import os
import math
import gc
from transformers import AutoTokenizer, AutoConfig
import torch 
import torch.nn as nn
import copy
from apex import amp
import wandb
import warnings
import itertools
warnings.filterwarnings("ignore", category=UserWarning)






def run(fold):
    
    
    # 1. Start a new run
    wandb.init(project="feedback", entity='surajsharan',config=config.wand_config(config))


    train_folds = pd.read_csv('input/train_folds.csv')
    train_folds.answers = train_folds.answers.apply(eval)
    
#     train_folds['context'] = train_folds['context'].apply(lambda x : " ".join(x.split()) )

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

    # test- training
    train_set, valid_set = train_folds[train_folds['kfold']
                                       != fold], train_folds[train_folds['kfold'] == fold]
    

    
    train_set.reset_index(drop=True,inplace=True)
    valid_set.reset_index(drop=True,inplace=True)
    
    
    valid_n_examples = valid_set.shape[0]
    
    del train_folds
    print(
        f'Training data : {train_set.shape} , Validation data :{valid_set.shape}')

    # pre processing data for Q/A
    train_features, valid_features = [], []

    for i, row in train_set.iterrows():
        train_features += preprocessing.get_features(row, tokenizer)

    for i, row in valid_set.iterrows():
        valid_features += preprocessing.get_features(row, tokenizer)

    # Data loaders
    train_dataloader = dataset.Datasetloader(features=train_features, test=False,inference=False).fetch(
        batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False)

    valid_dataloader = dataset.Datasetloader(features=valid_features, test=True,inference=False).fetch(
        batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False)

    
    del train_features ,train_set
    gc.collect()
    
    # Intialize Model
    device = config.DEVICE
    model_config = AutoConfig.from_pretrained(config.CONFIG_NAME)
    model = LongformerModel(config.MODEL_PATH, config=model_config)
    
    

    # optimizer
    optimizer = engine.get_optimizer(model, type="s")
    
    
    # mixed precision training with NVIDIA Apex
    if config.FP16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.FP16_OPT_LEVEL)
    
    
    if config.RUN_PARALLEL :
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = nn.DataParallel(model)
#         model = model.cuda() 
        
    model.to(device)
    
    # scheduler
    num_training_steps = math.ceil(
        len(train_dataloader) / config.GRADIENT_ACC_STEPS) * config.EPOCHS
    if config.WARMUP_RATIO > 0:
        num_warmup_steps = int(config.WARMUP_RATIO * num_training_steps)
    else:
        num_warmup_steps = 0
    print(
        f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")

    scheduler = engine.get_scheduler(
        optimizer, num_warmup_steps, num_training_steps)

    # Early stopping
    output_dir = os.path.join(config.PATH, f"checkpoint-fold-{fold}")

    early_stopping = engine.EarlyStopping(
        patience=config.EARLY_STOPPING, verbose=True, output_path=output_dir, tokenizer=tokenizer, model_config=model_config,metric="loss")
    
    

    wandb.watch(model)
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(
            train_dataloader, model, optimizer, device, scheduler)
        
        test_loss, start_logits , end_logits = engine.evaluate_modf(
            valid_dataloader, model, device)
        
        valid_features_ =  copy.deepcopy(valid_features)
        jaccard , valid_ans  = preprocessing.postprocess_qa_predictions(examples=valid_set, features=valid_features_,raw_predictions=(start_logits,end_logits) ,tokenizer = tokenizer, n_best_size = 20, max_answer_length = 40)
        del start_logits ,end_logits,valid_features_
        gc.collect()
        
        
        
        print(f"EPOCH : {epoch + 1}/{config.EPOCHS}")
        print(f"| Train Loss = {train_loss} | Valid Loss = {test_loss}")
        wandb.log({'EPOCH':epoch,'Train Loss': train_loss,'Valid Loss':test_loss}, commit=False)

        
        early_stopping(val_loss= test_loss, monitor_metric_val=jaccard,model=model)        
        if early_stopping.early_stop:
            print("Early stopping")
            break
  
    
    del model, tokenizer, model_config , train_dataloader,valid_dataloader , valid_set , valid_features
    gc.collect()
    


if __name__ == "__main__":
    
    for fold in range(2,5):  # (-1,5)
        print('-'*50)
        print(f'FOLD: {fold}')
        print('-'*50)
        run(fold)
        
        gc.collect()
    