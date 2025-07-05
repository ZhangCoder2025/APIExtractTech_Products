from re import T
#from tkinter.tix import Tree
import numpy as np
from regex import F
import torch
import random
import os
import argparse
from dataclasses import dataclass
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union
import json
from torch.cuda.amp import autocast
from sklearn.metrics import matthews_corrcoef

# read the parameters
def read_parameters():
    parser = argparse.ArgumentParser()
    
    # input file location
    parser.add_argument("--model_location",type=str,default="codebert-base/")
    
    parser.add_argument("--train_files_location",type=str,default="train_all.json")
    parser.add_argument("--valid_files_location",type=str,default="valid_all.json")
    parser.add_argument("--test_files_location",type=str,default="test_all.json")
    
    # model hyper parameters
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--n_epochs",type=int,default=30)
    parser.add_argument("--weight_decay",type=float,default=1e-2)
    parser.add_argument("--dropout",type=float,default=0.5)
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--optimizer',type=str,default='adamw')
    

    parser.add_argument("--bert_learning_rate",type=float,default=2e-5)
    parser.add_argument("--atten_learning_rate",type=float,default=2e-3)
    parser.add_argument("--learning_rate",type=float,default=2e-3)
    parser.add_argument("--classfier_learning_rate",type=float,default=2e-3)
    parser.add_argument("--charLSTM_learning_rate",type=float,default=2e-3)
    
    parser.add_argument('--context_embed_dim', type=int, default=256)

    parser.add_argument("--context_learning_rate",type=float,default=2e-5)

    parser.add_argument('--charLSTM_char_embed_dim', type=int, default=128)
    parser.add_argument('--charLSTM_hidden_size',type=int,default=200)
    parser.add_argument('--charLSTM_emb_dim', type=int, default=1024)
    parser.add_argument('--atten_dim', type=int, default=256)
    
    parser.add_argument("--loss_gamma",type=float,default=3)
    parser.add_argument("--loss_weight",type=float,default=0.7)
    
    parser.add_argument('--model_save_path', type=str, default="saved_models/")

    #device parameters
    parser.add_argument("--GPU_device",type=str,default="0")

    # char-level parameters
    parser.add_argument('--char_alphabet_size', type=int,default=70)

    args = parser.parse_args()

    return args

def preprocess_Datasets(raw_datasets):
    processed_dataset = raw_datasets
    if('full_names' in raw_datasets.column_names):
        processed_dataset = raw_datasets.remove_columns('full_names')
    if('mention_morphological' in raw_datasets.column_names):
        processed_dataset = processed_dataset.remove_columns('mention_morphological')
    if('mention_common' in raw_datasets.column_names):
        processed_dataset = processed_dataset.remove_columns('mention_common')
    if('api_common' in raw_datasets.column_names):
        processed_dataset = processed_dataset.remove_columns('api_common')

    return processed_dataset


'''
Hierarchical adjustment of learning rate
'''
def adjust_Hierarchical_lr(args,model):
    no_decay = ["bias", "LayerNorm.weight"]
    
    classifier_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.classfier_learning_rate},
        {'params': [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.classfier_learning_rate}
    ]

    context_param_optimizer = list(model.context.named_parameters())
    optimizer_grouped_parameters.append({'params': [p for n, p in context_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay, 'lr': args.context_learning_rate})
    optimizer_grouped_parameters.append({'params': [p for n, p in context_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.context_learning_rate})  

    bert_param_optimizer = list(model.bert.named_parameters())
    optimizer_grouped_parameters.append({'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.bert_learning_rate})
    optimizer_grouped_parameters.append({'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.bert_learning_rate})
 

    charlstm_param_optimizer = list(model.charLSTM.named_parameters())
    optimizer_grouped_parameters.append({'params': [p for n, p in charlstm_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay, 'lr': args.charLSTM_learning_rate})
    optimizer_grouped_parameters.append({'params': [p for n, p in charlstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': args.charLSTM_learning_rate})
    

    return optimizer_grouped_parameters



'''
dataloader
'''

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class SpeciDataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,

            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        word_ids = [feature['word_ids'] for feature in features]
        if padding_side == "right":
            batch['word_ids'] = [
                list(word_id) + [self.label_pad_token_id] * (sequence_length - len(word_id)) for word_id in word_ids
            ]
        else:
            batch['word_ids'] = [
                [self.label_pad_token_id] * (sequence_length - len(word_id)) + list(word_id) for word_id in word_ids
            ]

        chars = [feature['chars'] for feature in features]
        pad_chars = []
        if padding_side == "right":
            pad_chars = [chars[idx] + [[0]] * (sequence_length-len(chars[idx])) for idx in range(len(chars))]
        else:
            pad_chars = [[[0]] * (sequence_length-len(chars[idx])) + chars[idx] for idx in range(len(chars))]

        char_length_list = [list(map(len, pad_char)) for pad_char in pad_chars]

        batch_size = torch.tensor(batch["input_ids"]).shape[0]
        max_word_len = max(map(max, char_length_list))
        chars_tensor = torch.zeros((batch_size,sequence_length,max_word_len))

        char_seq_lengths = torch.tensor(char_length_list,dtype=torch.int64)
        if padding_side == "right":
            for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
                for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):

                    chars_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
        else:
            for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
                for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):

                    chars_tensor[idx, idy, (max_word_len-wordlen):] = torch.LongTensor(word)

        batch['chars'] = chars_tensor
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch['char_seq_lengths'] = char_seq_lengths
        return batch
    
    
    


"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""

class Alphabet:
    def __init__(self, name):
        self.name = name
        self.PAD = "[PAD]"
        self.UNKNOWN = "[UNK]"
        self.instance2index = {}
        self.instances = []

        self.next_index = 0
        
        self.add(self.PAD)
        self.add(self.UNKNOWN)
    

    def add(self, instance):
        if instance not in self.instance2index.keys():
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        try:
            return self.instances[index]
        except IndexError:
            print('WARNING:Alphabet Has No Such instance.')

    def size(self):
        return len(self.instances) 

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}





'''
computer experiment result with metrics
'''


def compute_middle_metrics(preds_list,labels_list):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pred, label in zip(preds_list,labels_list):
        if(pred=='B-API'):
            if(label=='B-API'):
                tp += 1
            else:
                fp += 1
        else:
            if(label=='0'):
                tn += 1
            else:
                fn += 1
    return tp,tn,fp,fn

def compute_metrics(label_list, raw_predicts,raw_labels):
    #all_true_predictions = []
    #all_true_labels = []
    all_tp = 0
    all_tn = 0
    all_fp = 0
    all_fn = 0
    for predictions,labels in zip(raw_predicts,raw_labels):
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        for bacth_pred,batch_label in zip(true_predictions,true_labels):
            batch_tp,batch_tn,batch_fp,batch_fn = compute_middle_metrics(bacth_pred,batch_label)
            all_tp += batch_tp
            all_tn += batch_tn
            all_fp += batch_fp
            all_fn += batch_fn
        
        #convert_true_predictions = list(_flatten(true_predictions))
        #convert_true_labels = list(_flatten(true_labels))
        #all_true_predictions.extend(convert_true_predictions)
        #all_true_labels.extend(convert_true_labels)
    #return metrics.classification_report(y_pred=all_true_predictions,y_true=all_true_labels,output_dict=True)
    if(all_tp == 0 and all_fp == 0 and all_fn == 0):
        real_precision = 1
        real_f1_score = 1
        real_recall = 1
    elif(all_tp == 0 and (all_fp > 0 or all_fn > 0)):
        real_precision = 0
        real_f1_score = 0
        real_recall = 0
    else:
        real_precision = all_tp / (all_tp + all_fp)
        real_recall = all_tp / (all_tp + all_fn)
        real_f1_score = 2 * real_precision * real_recall /(real_precision+real_recall)
    return real_precision, real_recall, real_f1_score



def train(model,train_dataloader,device,optimizer,lr_scheduler,progress_bar):
    model.train()
    for steps,batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        word_ids = batch.pop('word_ids')
        # moditified by crf
        with autocast():
            loss = model(**batch,is_train = True)
        # loss regularization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)




def validate(model, valid_dataloader, device):
    
    model.eval()
    all_tp = 0
    all_tn = 0
    all_fp = 0
    all_fn = 0
    token_num = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            word_ids = batch.pop('word_ids')
            with autocast():
                predicts = model(**batch,is_train = False)           
            if(torch.is_tensor(predicts)):
                predicts = predicts.to('cpu').numpy().flatten()
            else:
                predicts = sum(predicts,[])
            if(len(batch['labels'].to('cpu').numpy().flatten())!=len(predicts)):
                batch_labels = np.delete(batch['labels'].to('cpu').numpy().flatten(),np.where(batch['labels'].to('cpu').numpy().flatten()<0))
            else:
                batch_labels = batch['labels'].to('cpu').numpy().flatten()
            for single_predict,single_label in zip(predicts,batch_labels):
                token_num += 1
                if(single_label==1 and single_predict==1):
                    all_tp += 1
                elif(single_label==1 and single_predict==0):
                    all_fn += 1
                elif(single_label==0 and single_predict==0):
                    all_tn += 1
                elif(single_label==0 and single_predict==1):
                    all_fp += 1
    if(all_tp == 0 and all_fp == 0 and all_fn == 0):
        API_Precision = 1
        API_F1 = 1
        API_Recall = 1
    elif(all_tp == 0 and (all_fp > 0 or all_fn > 0)):
        API_Precision = 0
        API_F1 = 0
        API_Recall = 0
    else:
        API_Precision = all_tp / (all_tp + all_fp)
        API_Recall = all_tp / (all_tp + all_fn)
        API_F1 = 2 * API_Precision * API_Recall /(API_Precision+API_Recall)
    #API_Precision, API_Recall, API_F1 = compute_metrics(label_list=label_list, raw_predicts=raw_predicts,raw_labels=raw_labels)
    return model, API_Precision, API_Recall, API_F1








def test(model,eval_dataloader,device):
    model.eval()
    all_tp = 0
    all_tn = 0
    all_fp = 0
    all_fn = 0
    with torch.no_grad():
        
        all_predicts = []
        all_true_labels = []
        
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            word_ids = batch.pop('word_ids')
            with autocast():
                predicts = model(**batch,is_train = False)
            if(torch.is_tensor(predicts)):
                predicts = predicts.to('cpu').numpy().flatten()
            else:
                predicts = sum(predicts,[])
            if(len(batch['labels'].to('cpu').numpy().flatten())!=len(predicts)):
                batch_labels = np.delete(batch['labels'].to('cpu').numpy().flatten(),np.where(batch['labels'].to('cpu').numpy().flatten()<0))
            else:
                batch_labels = batch['labels'].to('cpu').numpy().flatten()
            
            all_predicts.append(predicts)
            all_true_labels.append(batch_labels)
            for single_predict,single_label in zip(predicts,batch_labels):
                if(single_label==1 and single_predict==1):
                    all_tp += 1
                elif(single_label==1 and single_predict==0):
                    all_fn += 1
                elif(single_label==0 and single_predict==0):
                    all_tn += 1
                elif(single_label==0 and single_predict==1):
                    all_fp += 1
    
    final_labels = np.concatenate(all_true_labels)
    process_labels = final_labels[np.where(final_labels>=0)]  
    final_predicts = np.concatenate(all_predicts)
    process_predicts = final_predicts[np.where(final_labels>=0)]
    
    API_MCC = matthews_corrcoef(y_true=process_labels,y_pred=process_predicts)
    if(all_tp == 0 and all_fp == 0 and all_fn == 0):
        API_Precision = 1
        API_F1 = 1
        API_Recall = 1
    elif(all_tp == 0 and (all_fp > 0 or all_fn > 0)):
        API_Precision = 0
        API_F1 = 0
        API_Recall = 0
    else:
        API_Precision = all_tp / (all_tp + all_fp)
        API_Recall = all_tp / (all_tp + all_fn)
        API_F1 = 2 * API_Precision * API_Recall /(API_Precision+API_Recall)
    return API_Precision, API_Recall, API_F1,API_MCC

