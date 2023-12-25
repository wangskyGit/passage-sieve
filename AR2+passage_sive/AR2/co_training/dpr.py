from transformers import BertModel, DPRConfig, DPRContextEncoder, DPRQuestionEncoder,BertConfig
import os

import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from AR2.sieve.loss import loss_my_cores,loss_my_cores_pos,loss_my_cores_positive_v2
from typing import Tuple
from torch import Tensor as T
from torch import nn
# sys.path.append('/home/student2020/ar2/AR2/utils')

from math import log
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        pooler_output = self.encoder(input_ids, attention_mask).pooler_output
        return pooler_output

    def save_pretrained(self, address):
        self.encoder.save_pretrained(address)
class HFBertEncoder2(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = False, **kwargs
    ) -> BertModel:
        #logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder2(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
        self.version = int(transformers.__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, path, dropout: float = 0.1, model_type=None):
        if model_type is None:
            model_type = path
        cfg = BertConfig.from_pretrained(model_type)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(model_type, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result = super().forward(**kwargs)
        sequence_output = result.last_hidden_state + 0 * result.pooler_output.sum()
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states
    
class BiEncoder2(torch.nn.Module):
    def __init__(self, args, check_point=False):
        ctx_model = args.ctx_model_path
        question_model = args.qry_model_path
        super().__init__()
        self.ctx_model = HFBertEncoder2.init_encoder(ctx_model)
        self.question_model = HFBertEncoder2.init_encoder(question_model)
        # config = DPRConfig()
        # config.vocab_size = self.ctx_model.config.vocab_size
        # context_encoder = DPRContextEncoder(config)
        # context_encoder.base_model.base_model.load_state_dict(self.ctx_model.state_dict(), strict=False)
        # query_encoder = DPRQuestionEncoder(config)
        # query_encoder.base_model.base_model.load_state_dict(self.question_model.state_dict(), strict=False)
        self.ck=check_point
        # self.ctx_model = context_encoder
        # self.question_model = query_encoder
        if check_point: 
            self.question_model = EncoderWrapper(self.question_model)
            self.ctx_model = EncoderWrapper(self.ctx_model)
        self.encoder_gpu_train_limit = args.encoder_gpu_train_limit
        self.device=args.device
    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        if not self.ck:
            _,output,_=model(input_ids=input_ids, attention_mask=attention_mask)
            return output
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)
    def evaluate(self,data,p_shape,mode='neg'):
        """
        p_shape: query num * negative num * dim
        """
        ctx_data={k: v.reshape((p_shape[0]*p_shape[1],p_shape[2])) for k, v in data['ctx'].items()}
        ctx_vector = self.encode(self.ctx_model, ctx_data)
        qry_vector = self.encode(self.question_model, data['qry'])## query num *  bert_dim
        #ctx_vector = ctx_vector.reshape((p_shape[0],p_shape[1],ctx_vector.shape[1]))# query num * negative num * bert_dim
        #r = torch.matmul(qry_vector, torch.transpose(ctx_vector, 0, 1))
        r=torch.cosine_similarity(qry_vector.unsqueeze(dim=1),ctx_vector.unsqueeze(dim=0),dim=-1)
        label=self.generate_label(p_shape).to(self.device)
        probs = F.log_softmax(r, dim=1)# query num * negative num
        
        if mode=='neg':
            loss_re = torch.mean(probs,dim=1)
            loss_sel = probs-loss_re.unsqueeze(dim=1)
            
        else:
            loss_re = (-1)*torch.mean(probs,dim=1) # confidence regularizer
            loss_sel =  (-1)*probs-loss_re.unsqueeze(dim=1)
        
        # hit1=(probs[:,-1]>log(1/p_shape[1])).sum()/p_shape[0]
        # hit2=(probs[:,-1]>probs.mean(dim=1)).sum()/p_shape[0]
        return r.detach().cpu().numpy(),probs.detach().cpu().numpy(),label.cpu().numpy()
    def generate_label(self,p_shape):
        label=torch.zeros((p_shape[0],p_shape[0]*p_shape[1]))
        for i in range(p_shape[0]):
            label[i,(i+1)*p_shape[1]-1]=1
        return label
    def generate_label2(self,p_shape):
        label=torch.zeros((p_shape[0],p_shape[0]*p_shape[1]))
        for i in range(p_shape[0]):
            label[i,(i+1)*p_shape[1]-1]=1
            label[i,(i+1)*p_shape[1]-2]=1
        return label    
    def forward(self, data, p_shape,epoch,mode='neg'):
        """
        p_shape: query num * negative num * dim
        """
        ctx_data={k: v.reshape((p_shape[0]*p_shape[1],p_shape[2])) for k, v in data['ctx'].items()}
        ctx_vector = self.encode(self.ctx_model, ctx_data)
        qry_vector = self.encode(self.question_model, data['qry'])## query num *  bert_dim
        #ctx_vector = ctx_vector.reshape((p_shape[0],p_shape[1],ctx_vector.shape[1]))# query num * negative num * bert_dim
        #r = torch.matmul(qry_vector, torch.transpose(ctx_vector, 0, 1))
       
        r=torch.cosine_similarity(qry_vector.unsqueeze(dim=1),ctx_vector.unsqueeze(dim=0),dim=-1)
        label=self.generate_label(p_shape).to(self.device)
        #sim=torch.cosine_similarity(ctx_vector,qry_vector.unsqueeze(1),dim=2)
        probs = F.log_softmax(r, dim=1)# query num * negative num
        loss,loss_v,sieve_score=loss_my_cores_positive_v2(epoch=epoch,probs=probs,y=label,device=self.device)
        # if mode=='neg':
        #     loss,loss_v,sieve_score=loss_my_cores(epoch=epoch,probs=probs,y=data['label'],device=self.device)
        # else:
        #     loss,loss_v,sieve_score=loss_my_cores_pos(epoch=epoch,probs=probs,y=data['label'],device=self.device)
        # tag = data['label'].where(data['label']==-1,1,0)
        # loss = F.nll_loss(probs.flatten(), tag.long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == p_shape[1]-1).sum() / p_shape[0]
        label2=self.generate_label2(p_shape)
        probs2=probs[label2.to(torch.bool)].reshape((p_shape[0],p_shape[1]))
    
        probs2 = F.log_softmax(probs2, dim=1)
        hit1=(probs[:,-1]>log(1/(p_shape[1]*p_shape[0]))).sum()/p_shape[0]
        hit2=(probs2[:,-1]>probs2.mean(dim=1)).sum()/p_shape[0]
        return loss, loss_v, sieve_score, accuracy, hit1, hit2

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.question_model.save_pretrained(addr + '/qry')

class BiEncoder(torch.nn.Module):
    def __init__(self, args, check_point=False):
        ctx_model = args.ctx_model_path
        question_model = args.qry_model_path
        super().__init__()
        self.ctx_model = HFBertEncoder.init_encoder(path=ctx_model)
        self.question_model = HFBertEncoder.init_encoder(path=question_model)
        # config = DPRConfig()
        # config.vocab_size = self.ctx_model.config.vocab_size
        # context_encoder = DPRContextEncoder(config)
        # context_encoder.base_model.base_model.load_state_dict(self.ctx_model.state_dict(), strict=False)
        # query_encoder = DPRQuestionEncoder(config)
        # query_encoder.base_model.base_model.load_state_dict(self.question_model.state_dict(), strict=False)
        self.ck=check_point
        # self.ctx_model = context_encoder
        # self.question_model = query_encoder
        if check_point: 
            self.question_model = EncoderWrapper(self.question_model)
            self.ctx_model = EncoderWrapper(self.ctx_model)
        self.encoder_gpu_train_limit = args.encoder_gpu_train_limit
        self.device=args.device
    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        if not self.ck:
            _,output,_=model(input_ids=input_ids, attention_mask=attention_mask)
            return output
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)
    def evaluate(self,data,p_shape,mode='neg'):
        """
        p_shape: query num * negative num * dim
        """
        ctx_data={k: v.reshape((p_shape[0]*p_shape[1],p_shape[2])) for k, v in data['ctx'].items()}
        ctx_vector = self.encode(self.ctx_model, ctx_data)
        qry_vector = self.encode(self.question_model, data['qry'])## query num *  bert_dim
        ctx_vector = ctx_vector.reshape((p_shape[0],p_shape[1],ctx_vector.shape[1]))# query num * negative num * bert_dim
        #dot_products =torch.sum(ctx_vector.mul(qry_vector.unsqueeze(1)),dim=2)# query num * negative num, 向量对应元素相乘再相加
        #sim=dot_products
        sim=torch.cosine_similarity(ctx_vector,qry_vector.unsqueeze(1),dim=2)
        probs = F.log_softmax(sim, dim=1)# query num * negative num
        if mode=='neg':
            loss_re = torch.mean(probs,dim=1)
            loss_sel = probs-loss_re.unsqueeze(dim=1)
            loss=probs
            loss_v=loss_sel<0
        else:
            loss_re = (-1)*torch.mean(probs,dim=1) # confidence regularizer
            loss_sel =  (-1)*probs-loss_re.unsqueeze(dim=1)
            loss=(-1)*probs
            loss_v=loss_sel>0 

        hit1=(probs[:,-1]>log(1/p_shape[1])).sum()/p_shape[0]
        hit2=(probs[:,-1]>probs.mean(dim=1)).sum()/p_shape[0]
        return loss.detach().cpu().numpy(),loss_v.detach().cpu().numpy(),hit1,hit2
    def forward(self, data, p_shape,epoch,mode='neg'):
        """
        p_shape: query num * negative num * dim
        """
        ctx_data={k: v.reshape((p_shape[0]*p_shape[1],p_shape[2])) for k, v in data['ctx'].items()}
        ctx_vector = self.encode(self.ctx_model, ctx_data)
        qry_vector = self.encode(self.question_model, data['qry'])## query num *  bert_dim
        ctx_vector = ctx_vector.reshape((p_shape[0],p_shape[1],ctx_vector.shape[1]))# query num * negative num * bert_dim
        #dot_products =torch.sum(ctx_vector.mul(qry_vector.unsqueeze(1)),dim=2)# query num * negative num, 向量对应元素相乘再相加
        #sim=dot_products
        
        sim=torch.cosine_similarity(ctx_vector,qry_vector.unsqueeze(1),dim=2)
        probs = F.log_softmax(sim, dim=1)# query num * negative num
        if mode=='neg':
            loss,loss_v,sieve_score=loss_my_cores(epoch=epoch,probs=probs,y=data['label'],device=self.device)
        else:
            loss,loss_v,sieve_score=loss_my_cores_pos(epoch=epoch,probs=probs,y=data['label'],device=self.device)
        # tag = data['label'].where(data['label']==-1,1,0)
        # loss = F.nll_loss(probs.flatten(), tag.long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == p_shape[1]-1).sum() / p_shape[0]
        hit1=(probs[:,-1]>log(1/p_shape[1])).sum()/p_shape[0]
        hit2=(probs[:,-1]>probs.mean(dim=1)).sum()/p_shape[0]
        return loss, loss_v, sieve_score, accuracy, hit1, hit2

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.question_model.save_pretrained(addr + '/qry')