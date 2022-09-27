## Strategies to try 
1) train 2 longformers based on length of the text  :- if the length is short try a smaller model(maybe longformer base or RoberTa) and if the length is large train with a longformer train a smaller one with 2036 and large with 4096
2) pre processing and post processing may help a lot in this modelling approach  
3) Impact on training :-pre processing : Cleaning the white space before tokenization , the discourse start- discourse end is not correct if we lift the text from the txt file there are white spaces
4) Post processing : play with th ethresholds of the discourse types which can help in smoothen the predictions 




# todo
0) Create data loaders for the model : done
1) Prepae a code base today to run the model over the weekend - for baseline can use roberta or longformers just to test the pipeline   : done  

2) Use a modified loss function

    def loss_fn(start_logits, end_logits,
            start_positions, end_positions):
    m = torch.nn.LogSoftmax(dim=1)
    loss_fct = torch.nn.KLDivLoss()
    start_loss = loss_fct(m(start_logits), start_positions)
    end_loss = loss_fct(m(end_logits), end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss



3) while post processing play with the best score in order to tune the answers (removing the junk characters say['.,!~'] some special characters from text)


4) Model architechture : multiple dropout 
     [multisample dropout (wut): https://arxiv.org/abs/1905.09788]
            logits = torch.mean(
                torch.stack(
                    [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            
    
5) apply this model architucture 
     Model-1
        Concat([last 2 hidden_layers from BERT]) -> Conv1D -> Linear
        End position depends on start (taken from here), which looks like,
        # x_head, x_tail are results after Conv1D
        logit_start = linear(x_start)
        logit_end = linear(torch.cat[x_start, x_end], dim=1)
        
        
