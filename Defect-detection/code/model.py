# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def _forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = self.dropout(outputs)
        
        logits = outputs  # Keep as raw logits
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            labels = labels.float()
            
            # Use built-in weighted BCE
            pos_weight = torch.tensor(getattr(self.args, 'pos_weight', 1.0)).to(labels.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], 
                labels, 
                pos_weight=pos_weight
            )
            
            return loss, prob
        else:
            return prob
          
from transformers import T5EncoderModel, T5Tokenizer

class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        # Same dropout as your current model
        self.dropout = nn.Dropout(getattr(args, 'dropout_probability', 0.1))
        
        # Classification head - adjust input size for CodeT5
        # CodeT5-base has 768 hidden size, same as CodeBERT
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, input_ids=None, labels=None):
        # Use only the encoder part of CodeT5
        encoder_outputs = self.encoder( input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)) 
        # Get the last hidden state
        outputs = encoder_outputs.last_hidden_state
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        # Pool the sequence (mean pooling over sequence length)
        # Mask padding tokens
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
        masked_outputs = outputs * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_outputs, dim=1)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = summed / lengths
        
        # Classification
        logits = self.classifier(pooled_output)
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            labels = labels.float()
            
            # Same loss as your current model
            pos_weight = getattr(self.args, 'pos_weight', 1.0)
            loss = torch.log(prob[:, 0] + 1e-10) * labels * pos_weight + torch.log((1-prob)[:, 0] + 1e-10) * (1-labels)
            loss = -loss.mean()
            
            return loss, prob
        else:
            return prob

# from https://github.com/saikat107/NatGen/blob/main/src/finetuning/models.py
class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        # outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
        #                        labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # hidden_states = outputs['decoder_hidden_states'][-1]
        
        hidden_states = outputs.last_hidden_state
        
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5' or self.args.model_type == 'natgen':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
