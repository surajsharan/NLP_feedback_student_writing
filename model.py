
from transformers import AutoModel
import torch
import torch.nn as nn
import config as configuration
import numpy as np
from sklearn import metrics


class LongformerModel(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(LongformerModel, self).__init__()
        self.config = config
        self.longformer = AutoModel.from_pretrained(
            modelname_or_path, config=config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(config.hidden_size, configuration.NUM_LABELS)
        

#         self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    
    
    def forward(self, ids, mask, word_ids = None,token_type_ids=None, targets=None):

        if token_type_ids:
            transformer_out = self.longformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.longformer(ids, mask)
            
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
#         logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = loss_func(logits1, targets, attention_mask=mask)
            loss2 = loss_func(logits2, targets, attention_mask=mask)
            loss3 = loss_func(logits3, targets, attention_mask=mask)
            loss4 = loss_func(logits4, targets, attention_mask=mask)
            loss5 = loss_func(logits5, targets, attention_mask=mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            
            return logits, loss

        return logits, loss


    
## ADAPTED

class XLM_RoBertamodel(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(XLM_RoBertamodel, self).__init__()
        self.config = config
        config.update({
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
            }) 
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config)
        
        
        
        self.high_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size * 2, 2)

        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                input_ids,
                attention_mask=None,
                start_position=None,
                end_position=None,
                offset_mapping=None,
                
                # token_type_ids=None
                ):
        
        
        out = self.xlm_roberta(input_ids,attention_mask=attention_mask,)
        LAST_HIDDEN_LAYERS = 12

        out = out.hidden_states 
        out = torch.stack(tuple(out[-i - 1] for i in range(LAST_HIDDEN_LAYERS)), dim=0)
       
        out_mean = torch.mean(out, dim=0) 
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1) 


        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([self.qa_outputs(self.high_dropout(out))for _ in range(5)], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

       
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if (start_position is not None) & (end_position is not None):
            loss = loss_fn((start_logits, end_logits),
                           (start_position, end_position))
            loss = loss / config.GRADIENT_ACC_STEPS

        return start_logits, end_logits, loss
    


def loss_func(outputs, targets, attention_mask):
    loss_fct = nn.CrossEntropyLoss()

    active_loss = attention_mask.view(-1) == 1
    active_logits = outputs.view(-1, configuration.NUM_LABELS)
    true_labels = targets.view(-1)
    outputs = active_logits.argmax(dim=-1)
    idxs = np.where(active_loss.cpu().numpy() == 1)[0]
    active_logits = active_logits[idxs]
    true_labels = true_labels[idxs].to(torch.long)

    loss = loss_fct(active_logits, true_labels)
    return loss   

    
