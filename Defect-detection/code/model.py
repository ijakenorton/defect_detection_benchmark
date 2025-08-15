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
        
        self.dropout = nn.Dropout(getattr(args, 'dropout_probability', 0.1))
        
        # Classification head - adjust input size for CodeT5
        # CodeT5-base has 768 hidden size, same as CodeBERT
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, input_ids=None, labels=None):
        # Use only the encoder part of CodeT5
        encoder_outputs = self.encoder( input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)) 
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

class CodeT5FullModel(nn.Module):
    """CodeT5 with full encoder-decoder + EOS token extraction"""
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5FullModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        self.dropout = nn.Dropout(getattr(args, 'dropout_probability', 0.1))
        self.classifier = nn.Linear(config.hidden_size, 1)
        
    def get_t5_vec(self, source_ids):
        """Extract representation from decoder hidden states using EOS tokens"""
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, 
                               output_hidden_states=True)
        
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)
        
        # Robust EOS handling
        if eos_mask.sum() == 0:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = source_ids.size(0)
            batch_indices = torch.arange(batch_size, device=source_ids.device)
            vec = hidden_states[batch_indices, seq_lengths, :]
        else:
            batch_size, seq_len = source_ids.shape
            vec_list = []
            
            for i in range(batch_size):
                eos_positions = torch.where(eos_mask[i])[0]
                
                if len(eos_positions) > 0:
                    last_eos_pos = eos_positions[-1]
                    vec_list.append(hidden_states[i, last_eos_pos, :])
                else:
                    seq_length = attention_mask[i].sum() - 1
                    vec_list.append(hidden_states[i, seq_length, :])
            
            vec = torch.stack(vec_list)
        
        return vec
        
    def forward(self, input_ids=None, labels=None):
        vec = self.get_t5_vec(input_ids)
        vec = self.dropout(vec)
        
        logits = self.classifier(vec)
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            labels = labels.float()
            pos_weight = torch.tensor(getattr(self.args, 'pos_weight', 1.0)).to(labels.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), labels, pos_weight=pos_weight
            )
            return loss, prob
        else:
            return prob

# And update the DefectModel to use the proper pos_weight:
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
        
        # Use the full T5 model with decoder
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, 
                               output_hidden_states=True)
        
        # Now we can access decoder_hidden_states
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        # Handle varying EOS tokens robustly
        if eos_mask.sum() == 0:
            # No EOS tokens - use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = source_ids.size(0)
            batch_indices = torch.arange(batch_size, device=source_ids.device)
            vec = hidden_states[batch_indices, seq_lengths, :]
        else:
            # Handle varying EOS tokens per sequence
            batch_size, seq_len = source_ids.shape
            vec_list = []
            
            for i in range(batch_size):
                eos_positions = torch.where(eos_mask[i])[0]
                
                if len(eos_positions) > 0:
                    last_eos_pos = eos_positions[-1]
                    vec_list.append(hidden_states[i, last_eos_pos, :])
                else:
                    seq_length = attention_mask[i].sum() - 1
                    vec_list.append(hidden_states[i, seq_length, :])
            
            vec = torch.stack(vec_list)
        
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
        
        if labels is not None:
            labels = labels.float()
            
            # Extract vulnerability logit (class 1 = vulnerable)
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.squeeze()
            
            # Use proper weighted BCE loss like other models
            pos_weight = torch.tensor(getattr(self.args, 'pos_weight', 1.0)).to(labels.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logits, 
                labels, 
                pos_weight=pos_weight
            )
            
            # Return probabilities in same format as other models
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return loss, prob
        else:
            # For inference
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.squeeze()
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return prob

# from https://github.com/saikat107/NatGen/blob/main/src/finetuning/models.py
class _DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # Handle EOS tokens more robustly
        eos_mask = source_ids.eq(self.config.eos_token_id)
        
        # Check if we have any EOS tokens at all
        if eos_mask.sum() == 0:
            # No EOS tokens found - use the last non-padding token for each sequence
            # Get the length of each sequence (number of non-padding tokens)
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because we want 0-indexed
            batch_size = source_ids.size(0)
            
            # Create indices for gathering
            batch_indices = torch.arange(batch_size, device=source_ids.device)
            vec = hidden_states[batch_indices, seq_lengths, :]
            
        else:
            # We have EOS tokens - handle the case where different sequences have different numbers
            batch_size, seq_len = source_ids.shape
            vec_list = []
            
            for i in range(batch_size):
                # Get EOS positions for this sequence
                eos_positions = torch.where(eos_mask[i])[0]
                
                if len(eos_positions) > 0:
                    # Use the last EOS token's representation
                    last_eos_pos = eos_positions[-1]
                    vec_list.append(hidden_states[i, last_eos_pos, :])
                else:
                    # No EOS in this sequence - use last non-padding token
                    seq_length = attention_mask[i].sum() - 1
                    vec_list.append(hidden_states[i, seq_length, :])
            
            vec = torch.stack(vec_list)
        
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
        prob = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
