import torch
import config
from preprocessing import target_id_map

# class DatasetRetriever(torch.utils.data.Dataset):
#     def __init__(self, samples, max_len ,test=False, inference=False):
#         super(DatasetRetriever, self).__init__()
#         self.samples = samples
#         self.max_len = max_len
#         self.inference = inference
#         self.test = test
        
        

#     def __len__(self):
#         return len(self.samples)

    
#     def __getitem__(self, item):
#         input_ids = self.samples[item]["input_ids"]
#         attention_mask = self.samples[item]["attention_mask"]
#         word_ids = self.samples[item]["word_ids"]
#         word_ids = [w if w is not None else config.NON_LABEL for w in word_ids]
        
#         if self.inference:
            
#             return {	
#                     "ids": input_ids,	
#                     "mask": attention_mask,	
#                     "word_ids" :word_ids
#                    } 
            
#         else:
#             input_labels = self.samples[item]["input_labels"]
#             pad_labels = ['PAD' for i in range(self.max_len)]
#             pad_labels[1:len(input_labels)] = input_labels[1:]
#             pad_labels = [target_id_map[x] for x in pad_labels]
            
            
#             return {
#                     "ids": torch.tensor(input_ids, dtype=torch.long),
#                     "mask": torch.tensor(attention_mask, dtype=torch.long),
#                     "targets": torch.tensor(pad_labels, dtype=torch.long),
#                     "word_ids" :torch.tensor(word_ids, dtype=torch.long)
#                     }



class DatasetRetriever(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_len ,test=False, inference=False):
        super(DatasetRetriever, self).__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.length = len(samples)
        self.inference = inference
        self.test = test
        
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        input_ids = self.samples[item]["input_ids"]
        
        if self.inference:
            
            # add start token id to the input_ids
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            
            if len(input_ids) > self.max_len - 1:	
                input_ids = input_ids[: self.max_len - 1]
                
            # add end token id to the input_ids	
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            
            attention_mask = [1] * len(input_ids)
            
            return {	
                    "ids": input_ids,	
                    "mask": attention_mask,	
                   } 
            
        else:
            input_labels = self.samples[item]["input_labels"]
            input_labels = [target_id_map[x] for x in input_labels]
            other_label_id = target_id_map["O"]
            padding_label_id = target_id_map["PAD"]
            
            # add start token id to the input_ids
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            input_labels = [other_label_id] + input_labels
            if len(input_ids) > self.max_len - 1:
                input_ids = input_ids[: self.max_len - 1]
                input_labels = input_labels[: self.max_len - 1]
            # add end token id to the input_ids
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            input_labels = input_labels + [other_label_id]
            attention_mask = [1] * len(input_ids)
            
            padding_length = self.max_len - len(input_ids)
            if padding_length > 0:
                if self.tokenizer.padding_side == "right":
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                    input_labels = input_labels + [padding_label_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                else:
                    input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                    input_labels = [padding_label_id] * padding_length + input_labels
                    attention_mask = [0] * padding_length + attention_mask
            
            return {
                    "ids": torch.tensor(input_ids, dtype=torch.long),
                    "mask": torch.tensor(attention_mask, dtype=torch.long),
                    "targets": torch.tensor(input_labels, dtype=torch.long),
                    }


class Datasetloader:
    def __init__(self, samples, tokenizer,max_len,test,inference,collate=None):
        self.samples = samples
        self.test = test
        self.inference= inference
        self.tokenizer = tokenizer 
        self.collate = collate
        self.max_len = max_len
#         self.dataset = DatasetRetriever(samples=self.samples, max_len=self.max_len ,test=self.test, inference=self.inference)
        self.dataset = DatasetRetriever(samples=self.samples,tokenizer=self.tokenizer, max_len=self.max_len ,test=self.test, inference=self.inference)

    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True):
        if not self.test:
            sampler = torch.utils.data.RandomSampler(self.dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(self.dataset)

        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last,collate_fn=self.collate)
        return data_loader
