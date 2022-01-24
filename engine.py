

import config
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from transformers import AdamW
from apex import amp
import os
from preprocessing import _get_jaccard_score
import collections
import wandb
import gc


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, output_path=None, tokenizer=None, model_config=None,metric="loss"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.metric = metric
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.counter = 0
        self.best_score = None
        self.monitor_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.loss_ = []
        self.monitor_ = []
        self.monitor_val_min = -np.Inf

    def __call__(self, val_loss,monitor_metric_val, model):

        
        score = -val_loss
        monitor_metric = monitor_metric_val
        self.loss_.append(score)
        self.monitor_.append(monitor_metric)

        if self.best_score is None:
            self.best_score = score
            self.monitor_score = monitor_metric
            
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            
        elif score > 1.2 * np.max(self.loss_) and monitor_metric == np.max(self.monitor_):
            self.best_score = score
            self.monitor_score = monitor_metric
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            
            
        elif score < self.best_score :
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
        else:
            self.best_score = score
            self.monitor_score = monitor_metric
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss,monitor_metric_val, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.metric == "loss":
                print( f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else :
                print( f'Jaccard score increased ({self.monitor_val_min:.6f} --> {monitor_metric_val:.6f}).  Saving model ...')
            
        os.makedirs(self.output_path, exist_ok=True)

        torch.save(model.state_dict(), f"{self.output_path}/pytorch_model.bin")
        self.model_config.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        
        self.val_loss_min = val_loss
        self.monitor_val_min = monitor_metric_val
        
        print(f"Saving model checkpoint to {self.output_path}.")
        
        
    def get_best_jaccard(self):
        return self.monitor_val_min 

# Metric Logger


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


def get_optimizer(model, type="s"):
    optimizer_parameters = get_optimizer_params(model, type="s")
    if config.OPTIMIZER == "AdamW":
        optimizer = AdamW(
            optimizer_parameters,
            lr=config.LEARNING_RATE,
            eps=config.EPSILON,
            correct_bias=True
        )
        return optimizer


def get_optimizer_params(model, type='s'):
    '''
    differential learning rate and weight decay
       s : unified lr for the whole model   
       i : differential lr for transformer and task layer
       a : differntial lr for transformer layer and task layers
    '''

    no_decay = ['bias', "LayerNorm.weight"]
    if type == 's':
        optimizer_parameters = filter(
            lambda x: x.requires_grad, model.parameters())
    elif type == 'i':
        optimizer_parameters = [
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': config.WEIGHT_DECAY},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    elif type == 'a':
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                     'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': config.WEIGHT_DECAY},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE / 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE * 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE / 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE * 2.6},
            {'params': [p for n, p in model.named_parameters() if config.MODEL_TYPE not in n],
             'lr': config.LEARNING_RATE * 20, "momentum": 0.99, 'weight_decay_rate': 0.0},
        ]
    return optimizer_parameters


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    if config.DECAY_NAME == "cosine-warmup":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return scheduler


def train_fn(dataloader, model, optimizer, device, scheduler):

    count = 0
    losses = AverageMeter()
    model.zero_grad()
    model.train()
    
    #Log gradients and model parameters
    wandb.watch(model)

    for batch_idx, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        for k, v in data.items():
            data[k] = v.to(device)

        _, _, loss = model(**data)

        
        if config.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.sum().backward()

        count += data['input_ids'].size(0)
        
        losses.update(loss.mean().item(), data['input_ids'].size(0))

        if batch_idx % config.GRADIENT_ACC_STEPS == 0 or batch_idx == len(dataloader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if batch_idx % config.LOGGING_STEPS == 0:
            # 4. Log metrics to visualize performance
            wandb.log({"train loss": loss.mean().item()})

    return losses.avg


def evaluate(dataloader, model, device, inference=False):
    losses = AverageMeter()
    model.eval()
    predictions, true_vals = [], []
    with torch.no_grad():
        
        if inference:
            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)

                start_logits, end_logits, _ = model(**data)
                predictions.append([output.detach().cpu().numpy()
                                   for output in (start_logits, end_logits)])

        else:
            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                    
                start_logits, end_logits, loss = model(**data)
                losses.update(loss.mean().item(), data['input_ids'].size(0))
                
                true_vals.append([output.detach().cpu().numpy()
                                   for output in (data['start_position'], data['end_position'])])
                predictions.append([output.detach().cpu().numpy()
                                   for output in (start_logits, end_logits)])

    return losses.avg, predictions, true_vals


def evaluate_modf(dataloader, model, device,inference=False):
    losses = AverageMeter()
    model.eval()
    jaccard_score = 0.0
    start_logits ,end_logits = [],[]
    wandb.watch(model)
    with torch.no_grad():
        
        if inference:
            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)

                start_logits, end_logits, _ = model(**data)
                predictions.append([output.detach().cpu().numpy()
                                   for output in (start_logits, end_logits)])

        else:
            for batch_idx, data in enumerate(tqdm(dataloader, total=len(dataloader))):
                for k, v in data.items():
                    if isinstance(v,list):
                        data[k] = v
                    else:
                        data[k] = v.to(device)
                    
                outputs_start, outputs_end, loss = model(data['input_ids'],data['attention_mask'],data['start_position'],data['end_position'])
                losses.update(loss.mean().item(), data['input_ids'].size(0))
                
                
                start_logits.append(outputs_start.cpu().numpy().tolist())
                end_logits.append(outputs_end.cpu().numpy().tolist())
                
                del outputs_start,outputs_end
                gc.collect()
                
                if batch_idx % config.LOGGING_STEPS == 0:
                    # 4. Log metrics to visualize performance
                    wandb.log({"valid loss": loss.mean().item()})
                            
    return losses.avg, np.vstack(start_logits), np.vstack(end_logits)




