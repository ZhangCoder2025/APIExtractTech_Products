from ast import arg
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel,BertModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from losses import FocalLoss
import numpy as np
from transformers import AutoTokenizer
import io
from typing import List, Optional


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Attention_CharBiLSTM(nn.Module):

    def __init__(self,
                 alphabet_size,
                 char_embed_dim = 150,
                 hidden_size = 200,
                 embed_dim = 150,
                 dropout = 0,
                 need_char_lstm_atten = True,
                 pool_method = 'max',
                 activation = 'relu'):
        super(Attention_CharBiLSTM, self).__init__()


        #embed(x)
        self.char_embedding = nn.Embedding(alphabet_size, char_embed_dim)

        self.char_embedding.weight.data.copy_(
            torch.from_numpy(self.random_embedding(alphabet_size, char_embed_dim)))

        # Dropout(x)
        self.dropout = nn.Dropout(dropout)
        # LSTM(x)
        self.bilstm = nn.LSTM(char_embed_dim, hidden_size // 2, num_layers=1, batch_first=True,bidirectional=True)

        self.w = nn.Parameter(torch.randn(hidden_size))
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.need_char_lstm_atten = need_char_lstm_atten
        if(need_char_lstm_atten==False):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
            self.pool_method = pool_method


    def random_embedding(self,vocab_size, embedding_dim):
        pretrain_emb = np.zeros([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(0, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, chars,char_seq_lengths): # chars.shape: tensor (batch_size, sequence_len, max_word_len)
        batch_size = chars.shape[0]
        sequence_length = chars.shape[1]
        max_word_len = chars.shape[2]

        chars_masks = chars.eq(0)


        # embed(x)
        chars = self.char_embedding(chars)

        # Dropout(x)
        chars = self.dropout(chars)


        reshape_chars = chars.view(batch_size*sequence_length,max_word_len,-1)
        reshape_char_seq_lengths = char_seq_lengths.view(batch_size*sequence_length,)

        # sort the char to input the LSTM
        reshape_char_seq_lengths, char_perm_idx = reshape_char_seq_lengths.sort(0, descending=True)
        reshape_chars = reshape_chars[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)

        # LSTM(x)
        pack_input = pack_padded_sequence(reshape_chars, reshape_char_seq_lengths.cpu(), batch_first=True)
        self.bilstm.flatten_parameters()
        lstm_output, lstm_hidden = self.bilstm(pack_input)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True, total_length=max_word_len)

        lstm_output = lstm_output[char_seq_recover]



        if(self.need_char_lstm_atten):
            M = torch.tanh(lstm_output)
            att_score = torch.matmul(M,self.w)
            reshape_char_masks = chars_masks.view(batch_size*sequence_length,-1)
            #if(att_score.view(-1)[torch.argmin(att_score)].item()  <= -1):
            att_score = att_score.masked_fill(reshape_char_masks, float('-inf'))
            att_weight = F.softmax(att_score, dim=1).unsqueeze(-1)
            att_weight = torch.where(torch.isnan(att_weight), torch.full_like(att_weight, 0), att_weight)
            if(torch.isnan(att_weight).any()):
                print("\n")
                print("values not satisfied")
                os.system("pause")
            out = lstm_output.mul(att_weight)
            out = torch.sum(out,dim=1)
            out = torch.tanh(out)
            out = self.fc(out)
            chars = out.view(batch_size,sequence_length,-1)
        else:
            lstm_chars = lstm_hidden[0].transpose(1,0).contiguous()[char_seq_recover]
            lstm_chars = lstm_chars.view(batch_size,sequence_length,-1)
            chars = self.fc(lstm_chars)

        return self.dropout(chars)





class CAREER(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config,args):
        super(CAREER, self).__init__(config)

        self.char_alphabet_size = args.char_alphabet_size
        self.feature_dim = 0
        self.context_hidden_dim = args.context_embed_dim
        self.loss_gamma = args.loss_gamma
        self.loss_weight = args.loss_weight
        '''
        multi-level word feature
        '''

        # word-level

        self.feature_dim += config.hidden_size

        self.bert = BertModel(config,add_pooling_layer=False)

        # global char-level feature

        self.charLSTM_char_embed_dim =args.charLSTM_char_embed_dim
        self.charLSTM_emb_dim = args.charLSTM_emb_dim
        self.charLSTM_dropout = args.dropout
        self.charLSTM_hidden_size = args.charLSTM_hidden_size
        self.feature_dim += self.charLSTM_emb_dim

        self.charLSTM = Attention_CharBiLSTM(
        alphabet_size = self.char_alphabet_size,
        embed_dim = self.charLSTM_emb_dim,
        char_embed_dim = self.charLSTM_char_embed_dim,
        hidden_size=self.charLSTM_hidden_size,
        dropout = self.charLSTM_dropout)

        #context encoder
        self.context = nn.LSTM(self.feature_dim,  self.context_hidden_dim // 2, dropout=args.dropout, batch_first=True,bidirectional=True)

        # classifier
        self.classifier = nn.Linear(self.context_hidden_dim, config.num_labels)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

        # init weights
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        tokenizer = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        chars = None,
        char_seq_lengths = None,
        is_train = True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        embeds = []

        bert_outputs =self.bert(
        input_ids = input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_hidden_states = output_hidden_states)

        word_hidden_states = bert_outputs['hidden_states']
        word_features = word_hidden_states[-1]
        embeds.append(word_features)

        char_lstm_features = self.charLSTM(chars,char_seq_lengths)
        embeds.append(char_lstm_features)

        embeds = torch.cat(embeds, 2)

        sequence_output = self.dropout(embeds)

        max_sequence_length = sequence_output.shape[1]
        word_seq_lengths = torch.LongTensor([torch.count_nonzero(word_seq).item()  for word_seq in input_ids])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        sequence_output = sequence_output[word_perm_idx]
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)

        pack_input = pack_padded_sequence(sequence_output,word_seq_lengths.cpu(),batch_first=True)
        self.context.flatten_parameters()
        context_output, context_hidden = self.context(pack_input)
        context_output, _ = pad_packed_sequence(context_output, batch_first=True, total_length=max_sequence_length)
        context_output = context_output[word_seq_recover]

        sequence_output = self.dropout(context_output)

        logits = self.classifier(sequence_output)

        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        logits = logits.view(batch_size * seq_len, -1)
        if(is_train):
            loss_function = FocalLoss(ignore_index=-100,gamma=self.loss_gamma,weight=torch.tensor((self.loss_weight,1-self.loss_weight)).to(device='cuda'))
            loss = loss_function.forward(input=logits,target=labels.view(batch_size * seq_len))
            return loss
        else:
            score = F.log_softmax(logits, 1)
            _, tags  = torch.max(score, 1)
            tags = tags.view(batch_size, seq_len)
            return tags




