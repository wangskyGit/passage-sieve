import sys
import os
sys.path.append('./')


import random
from transformers import BertModel, BertTokenizer,BertConfig
from torch.utils.data import DataLoader,Dataset,RandomSampler,DistributedSampler
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from dpr import BiEncoder
#from datasets import Dataset
import json
import os
import collections
import numpy as np
import argparse
from utils.util import normalize_passage,normalize_question,_normalize
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    all_gather_list
)
CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'epoch'])
from tqdm import tqdm
import torch.distributed as dist

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--global_step",
        default=0,
        type=int,
        help='global step'
    ) 
    parser.add_argument(
        "--mode",
        default='neg',
        type=str,
        help='neg or pos mode'
    ) 
    parser.add_argument(
        "--seed",
        default=10086,
        type=int,
        help='random seed'
    ) 
    parser.add_argument(
        "--renew",
        default=False,
        type=bool,
        help='whether renew the eval dataloader'
    ) 
    parser.add_argument(
        "--log_dir",
        default='./tb_log',
        type=str,
        help='tesorboard log dir'
    ) 
    parser.add_argument(
        "--shuffle_positives",
        default=True,
        type=bool,
        help='whether shuffle positives'
    )
    parser.add_argument(
        "--num_hard_negatives",
        default=5,
        type=int,
        help='number of hard negatives during training'
    )
    parser.add_argument(
        "--num_negatives_eval",
        default=5,
        type=int,
        help='number of candidates hard negatives during inferencing'
    )

    parser.add_argument(
        "--model_path",
        default='~/DPR/outputs/2023-05-31/08-33-38/output/dpr_biencoder.15',
        type=str,
        help='Model path for the initial retrieval encoder model'
    )
    parser.add_argument(
        "--ctx_model_path",
        default='nghuyong/ernie-2.0-base-en',
        type=str,
        help='Model name for context encoder model'
    )
    parser.add_argument(
        "--qry_model_path",
        default='nghuyong/ernie-2.0-base-en',
        type=str,
        help='Model name for qry encoder model'
    )
    parser.add_argument(
        "--path_to_dataset",
        default='./NQ/train_ce_0.json',
        type=str,
        help='The path of dataset'
    )
    parser.add_argument(
        "--path_save_model",
        default='./model/',
        type=str,
        help='The path for saving finetuning model'
    )
    parser.add_argument(
        "--valid",
        default=False,
        type=bool,
        help='Exist validation set or not'
    )
    parser.add_argument(
        "--test",
        default=False,
        type=bool,
        help='Exist test set or not'
    )
    parser.add_argument(
        "--encoder_gpu_train_limit",
        default=50,
        type=int,
        help='The number of samples that the gpu can put each time'
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help='Batchsize for training and evaluation'
    )
    parser.add_argument(
        "--lr",
        default=1e-7,
        type=float,
        help='Learning rate for training'
    )
    parser.add_argument(
        "--epoch",
        default=1,
        type=int,
        help='Training epoch number'
    )
    parser.add_argument(
        "--has_cuda",
        default=True,
        type=bool,
        help='Has cuda or not'
    )
    #===============================================================================================
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    return args


def load_data(args):
    file_addr = args.path_to_dataset
    domains = ['train']
    if args.valid:
        domains.append('valid')
    if args.test:
        domains.append('test')
    all_files = os.listdir(file_addr)
    concrete_dataset = []
    concrete_addr = []
    for every_split_domain in domains:
        for every_file in all_files:
            if every_split_domain in every_file and 'json' in every_file:
                concrete_addr.append(file_addr + '/' + every_file)
                break
    for every_file_addr in concrete_addr:
        file_in = open(every_file_addr, 'r')
        data = json.load(file_in)
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        print("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
        print("Total cleaned data size: {}".format(len(pre_data)))
        all_query = []
        all_negatives = []
        all_answer = []
        all_positive=[]
        for sample in pre_data:
            # if len(every_dialog['hard_negative_ctxs']):
            #     now_negative = random.choice(every_dialog['hard_negative_ctxs'][:15])
            # else:
            #     now_negative = random.choice(every_dialog['negative_ctxs'][:15])
            q=sample['question']
            ans=sample['answers']
            hard_negative_passages = sample['hard_negative_ctxs']
            positive_passages = sample['positive_ctxs']
            if isinstance(hard_negative_passages,dict):
                hard_negative_passages=[hard_negative_passages]
            random.shuffle(hard_negative_passages)
            if len(hard_negative_passages) < args.num_hard_negatives:
                hard_negative_passages = hard_negative_passages*args.num_hard_negatives
            hard_neg_ctxs = hard_negative_passages[0:args.num_hard_negatives]
            if args.shuffle_positives:
                positive_passagese_ctx = random.choice(positive_passages)
            else:
                positive_passagese_ctx = positive_passages[0]
            all_query.append(q)
            all_answer.append(ans)
            all_negatives.append(hard_neg_ctxs)
            all_positive.append(positive_passagese_ctx)
        build_dataset = {
            'query': all_query,
            'negative': all_negatives,
            'positive': all_positive,
            'answer': all_answer
        }
        concrete_dataset.append(build_dataset)
    return concrete_dataset
class TraditionDataset(Dataset):
    def __init__(self, file_path, tokenizer,num_hard_negatives=1, is_training=True,
            max_seq_length =256 ,max_q_length=32,shuffle_positives=False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.num_queries=len(self.data)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_seq_length = max_seq_length
        self.max_q_length = max_q_length
        self.shuffle_positives = shuffle_positives
    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            print('Aggregated data size: {}'.format(len(data)))
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        print("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
        print("Total cleaned data size: {}".format(len(pre_data)))
        return pre_data

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = normalize_question(json_sample["question"])
        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )
        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None
        positive_passages = positive_ctxs
        hard_negative_passages = hard_negative_ctxs
        if isinstance(hard_negative_passages,dict):
            hard_negative_passages=[hard_negative_passages]
        if len(hard_negative_passages) < self.num_hard_negatives:
            hard_negative_passages = hard_negative_passages*self.num_hard_negatives
        index_list=list(range(len(hard_negative_passages)))
        if self.is_training:
            random.shuffle(index_list)
        hard_index_list=index_list[0:self.num_hard_negatives]
        hard_neg_ctxs = [hard_negative_passages[i] for i in hard_index_list]
        if self.shuffle_positives:
            positive_passagese_ctx = [random.choice(positive_passages)]
        else:
            positive_passagese_ctx = positive_passages
        ### phase1:here to add negative retrieval and give confidence hard negative
        pos_token_ids=[self.tokenizer.encode(ctx['title'], text_pair=ctx['text'].strip(), add_special_tokens=True,
                                        max_length=self.max_seq_length,truncation=True,
                                        pad_to_max_length=False) for ctx in positive_passagese_ctx]
        neg_token_ids = [self.tokenizer.encode(ctx['title'], text_pair=ctx['text'].strip(), add_special_tokens=True,
                                        max_length=self.max_seq_length,truncation=True,
                                        pad_to_max_length=False) for ctx in hard_neg_ctxs ]
        question_token_ids = self.tokenizer.encode(query)
        answers = [self.tokenizer.encode(_normalize(single_answer),add_special_tokens=False) for single_answer in json_sample['answers']]
        return question_token_ids,pos_token_ids,neg_token_ids,answers,hard_index_list,index
    
    def __len__(self):
        return len(self.data)
    @classmethod
    def get_collate_fn(cls,num_negatives):
        def create_biencoder_input(features):
            q_list = []
            p_list = []
            n_list=[]
            for index, feature in enumerate(features):
                q_list.append(feature[0]) 
                p_list.extend(feature[1])
                n_list.extend(feature[2])
            d_list=p_list+n_list
            max_q_len = max([len(q) for q in q_list])
            max_d_len = max([len(d) for d in d_list])
            q_list = [q+[0]*(max_q_len-len(q)) for q in q_list]
            d_list = [d+[0]*(max_d_len-len(d)) for d in d_list]
            q_tensor = torch.LongTensor(q_list)
            doc_tensor = torch.LongTensor(d_list)
            p_tensor=doc_tensor[0:len(p_list)]# # query_num * vec_dim
            n_tensor=doc_tensor[len(p_list):].reshape((len(q_list),num_negatives,max_d_len))# query_num * neg_num * vec_dim
            q_num,d_num = len(q_list),len(d_list)
            return {
                    'query': [q_tensor,(q_tensor!= 0).long()],
                    'positive': [p_tensor,(p_tensor!=0).long()],
                    'negative':[n_tensor,(n_tensor!=0).long()],
                    "answers": [feature[3] for feature in features],
                    "hn_index":[feature[4] for feature in features],
                    "index":[feature[5] for feature in features]
                 }
        return create_biencoder_input

def set_env(args):
    # Setup CUDA, GPU & distributed training
    print('==================set env=================')
    if args.local_rank == -1 or not args.has_cuda:
        device = torch.device("cuda:2" if torch.cuda.is_available() and args.has_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        #torch.distributed.init_process_group(backend="nccl")
        print(args.local_rank,device)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        print(args.local_rank,device)
    args.device = device
    set_seed(args)
def load_states_from_checkpoint_ict(model_file: str) -> CheckpointState:
    from torch.serialization import default_restore_location
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    new_stae_dict = {}
    for key, value in state_dict['model']['query_model']['language_model'].items():
        new_stae_dict['question_model.' + key] = value
    for key, value in state_dict['model']['context_model']['language_model'].items():
        new_stae_dict['ctx_model.' + key] = value
    return new_stae_dict
def load_model(args):
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
        
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.ctx_model_path)
    # qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.qry_model_path)
    saved_state = load_states_from_checkpoint(args.model_path)
    model = BiEncoder(args)
    model.load_state_dict(saved_state.model_dict,strict=False)
    model.to(args.device)
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True)
    if args.local_rank == 0:
        torch.distributed.barrier() 
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=False,
        )

    return model,tokenizer
def main_work():
    args = get_arguments()
    #args.local_rank=int(os.environ['LOCAL_RANK'])
    set_env(args)
    device=args.device
    # data = load_data(args)
    # build_train_dataset = data[0]
    # train_dataset = Dataset.from_dict(build_train_dataset)
    model,tokenizer=load_model(args)
    if args.local_rank == 0:
        print(args)
    if args.local_rank != -1:
        dist.barrier()
    train_dataset =TraditionDataset(file_path=args.path_to_dataset,tokenizer=tokenizer,num_hard_negatives=args.num_hard_negatives,
                                    shuffle_positives=True,max_seq_length=256
                                    )
    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset,shuffle=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=TraditionDataset.get_collate_fn(args.num_hard_negatives),sampler=train_sample,batch_size=args.batch_size, shuffle=False)
    # if args.valid:
    #     build_dev_dataset = data[1]
    #     dev_dataset = Dataset.from_dict(build_dev_dataset)
    #     dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    #     dev_batch_size = args.batch_size
    tb_writer = None
    if is_first_worker():
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    optimizer = AdamW(model.parameters(), lr=args.lr,betas=(0.8,0.9))
    num_epochs = args.epoch
    num_training_steps = num_epochs * train_dataset.num_queries // args.batch_size
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,num_warmup_steps=0.2 * num_training_steps,
        num_training_steps=num_training_steps
    )
    model.zero_grad()
    model.train()
    save_base_directory = args.path_save_model
    loss_v_total=np.ones((train_dataset.num_queries,args.num_hard_negatives+1))
    hn_index=np.zeros((train_dataset.num_queries,args.num_hard_negatives))
    global_step=0
    loss_total=np.zeros((train_dataset.num_queries,args.num_hard_negatives+1))
    #global_step=0
    for epoch in range(num_epochs):
        raw_index=[]
        whole_loss = 0
        whole_num = 0
        cur_index=0
        for index, sample in enumerate(tqdm(train_dataloader)):
            raw_index+=sample['index']
            query_num=len(sample['query'][0])
            hn_index[cur_index:cur_index+query_num,:]=sample['hn_index']
            label = torch.tensor(([-1]*args.num_hard_negatives+[1])*query_num).to(device)
            ctx_ids=torch.cat((sample['negative'][0],sample['positive'][0].unsqueeze(1)),dim=1)
           
            ctx_mask=torch.cat((sample['negative'][1],sample['positive'][1].unsqueeze(1)),dim=1)
            qry = {"input_ids": sample['query'][0].long().to(device),
                          "attention_mask": sample['query'][1].long().to(device)
                          }
            ctx = {"input_ids": ctx_ids.long().to(device),
                          "attention_mask": ctx_mask.long().to(device)
                          }
            batch = {'ctx': ctx, 'qry': qry, 'label': label}
            loss, loss_v, sieve_score,accuracy,hit1,hit2 = model(batch,ctx_ids.shape,epoch,mode=args.mode)
            loss_v_total[cur_index:cur_index+query_num,:]=loss_v
            loss_total[cur_index:cur_index+query_num,:]=sieve_score.reshape((query_num,args.num_hard_negatives+1))
            loss.backward()
            whole_num += 1
            whole_loss += loss
            temp_v=loss_v_total[:cur_index+query_num,:]
            sieve_rate=1-temp_v.sum()/(temp_v.size)
            if index % 100 == 0 and args.local_rank in [-1,0]:
                tqdm.write('{} loss: {}'.format(whole_num,whole_loss.cpu().detach().numpy() / whole_num))
                tqdm.write(f'lr: {lr_scheduler.get_last_lr()}, accuracy: {accuracy.cpu().detach().numpy() },over-average-positive rate:{hit1.cpu().detach().numpy() },{hit2.cpu().detach().numpy() }')
                tqdm.write('sieved out rate:{}'.format(sieve_rate))
            if index % 10==0 and args.local_rank in [-1,0]:
                logs = {}
                logs["learning_rate"] = lr_scheduler.get_last_lr()[0]
                logs["loss"] = whole_loss.cpu().detach().numpy()  / whole_num
                logs["hit1"] = hit1.cpu().detach().numpy() 
                logs["hit2"] = hit2.cpu().detach().numpy() 
                logs['sieve_rate']=sieve_rate
                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
            global_step+=1
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            cur_index+=query_num
            torch.cuda.empty_cache()
        pt_save_directory = save_base_directory + str(args.global_step)
        loss_v_neg=loss_v_total[:,:-1]
        sieve_score_neg=loss_total
        if args.local_rank in [-1,0]:
            if not os.path.exists(pt_save_directory): #判断所在目录下是否有该文件名的文件夹
                os.mkdir(pt_save_directory)
            np.save(pt_save_directory + '/lossV',loss_v_neg)
            np.save(pt_save_directory + '/sieveScore',sieve_score_neg)
            np.save(pt_save_directory + '/hnIndex',hn_index)
            np.save(pt_save_directory + '/rawIndex',raw_index)
        if args.local_rank != -1:
            dist.barrier()
        
    tqdm.write("=======final evaluation=======")
    file_path=args.path_to_dataset
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        print('Aggregated data size: {}'.format(len(data)))
    # filter those without positive ctx
    pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
    print("cleaned data size: {} after positive ctx".format(len(pre_data)))
    pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
    print("Total cleaned data size: {}".format(len(pre_data)))
    pre_data=pre_data[:]
    if args.renew:
        
        train_dataset =TraditionDataset(file_path=args.path_to_dataset,tokenizer=tokenizer,num_hard_negatives=args.num_negatives_eval,
                                    shuffle_positives=True,max_seq_length=256
                                    )
        train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset,shuffle=False)
        train_dataloader = DataLoader(train_dataset, collate_fn=TraditionDataset.get_collate_fn(args.num_negatives_eval),sampler=train_sample,batch_size=args.batch_size, shuffle=False)
        num_negatives_eval=args.num_negatives_eval
    else:
        num_negatives_eval=args.num_hard_negatives
    raw_index=[]
    whole_loss = 0
    whole_num = 0
    cur_index=0
    model.eval()
    loss_v_total=np.ones((train_dataset.num_queries,num_negatives_eval+1))
    hn_index=np.zeros((train_dataset.num_queries,num_negatives_eval))
    loss_total=np.zeros((train_dataset.num_queries,num_negatives_eval+1))
    with torch.no_grad():
        for index, sample in enumerate(tqdm(train_dataloader)):
            raw_index=sample['index']
            query_num=len(sample['query'][0])
            hn_index=np.array(sample['hn_index'])
            label = torch.tensor(([-1]*num_negatives_eval+[1])*query_num).to(device)
            ctx_ids=torch.cat((sample['negative'][0],sample['positive'][0].unsqueeze(1)),dim=1)
            ctx_mask=torch.cat((sample['negative'][1],sample['positive'][1].unsqueeze(1)),dim=1)
            qry = {"input_ids": sample['query'][0].long().to(device),
                            "attention_mask": sample['query'][1].long().to(device)
                            }
            ctx = {"input_ids": ctx_ids.long().to(device),
                            "attention_mask": ctx_mask.long().to(device)
                            }
            batch = {'ctx': ctx, 'qry': qry, 'label': label}
            loss_sel, loss_v,hit1,hit2 = get_model_obj(model).evaluate(batch,ctx_ids.shape,mode=args.mode)
            loss_v_total[cur_index:cur_index+query_num,:]=loss_v
            loss_total[cur_index:cur_index+query_num,:]=loss_sel.reshape((query_num,num_negatives_eval+1))
            temp_v=loss_v_total[:cur_index+query_num,:]
            sieve_rate=1-temp_v.sum()/(temp_v.size)
            loss_v_neg=loss_v[:,:-1]
            for i in range(len(raw_index)):
                sample_i=pre_data[raw_index[i]]
                negs=sample_i["hard_negative_ctxs"]
                v=loss_v_neg[i,:].astype(bool)
                if len(negs) < num_negatives_eval:
                        negs = negs*num_negatives_eval
                if sum(v)==0:
                    v[random.randint(0,len(v)-1)]=True
                if len(negs)<max(hn_index[i,:][v]):   
                    print(i) 
                    print(len(negs))
                    print(hn_index[i,:][v])
                    continue
                sample_i["hard_negative_ctxs"]=[negs[int(k)] for k in hn_index[i,:][v]]
            if index % 100 == 0 and args.local_rank in [-1,0]:
                tqdm.write(f'lr: {lr_scheduler.get_last_lr()},over-average-positive rate:{hit1.cpu().detach().numpy() },{hit2.cpu().detach().numpy() }')
                tqdm.write('sieved out rate:{}'.format(sieve_rate))
            if index % 10==0 and args.local_rank in [-1,0]:
                logs = {}
                logs["hit1"] = hit1.cpu().detach().numpy() 
                logs["hit2"] = hit2.cpu().detach().numpy() 
                logs['sieve_rate']=sieve_rate
                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
            global_step+=1
            cur_index+=query_num
            
        
        if args.local_rank != -1:
            dist.barrier()
    
    if args.local_rank != -1:
        print('generate new sieved datasets')
        with open(file_path+str(args.local_rank),'w') as f:
            json.dump(sieved_dataset,f)
        dist.barrier()
    if args.local_rank == 0:
        agg_dataset=[]
        for i in range(args.world_size):
            with open(file_path+str(i),'r') as f:
                agg_dataset+=json.load(f)
        print('total len of aggregated dataset:',len(agg_dataset))
        with open(file_path,'w') as f:
            json.dump(agg_dataset,f)

    if args.local_rank != -1:
        dist.barrier()
    
if __name__ == "__main__":
    main_work()
