# -*- encoding: utf-8 -*-
'''
@Time    :   2021/09/10 22:36:39
@Author  :   流氓兔233333 
@Version :   1.0
@Contact :   squad 'version'
'''

import json
import numpy as np
from tqdm import tqdm
import pickle, os
import random

# load data
# with open('./data_raw/train-v2.0.json','r',encoding='utf8')as fp:
#     json_data = json.load(fp)

# data_QA = json_data['data']
# data_prepare = []
# for paragraphs in tqdm(data_QA):
#     for paragraph in tqdm(paragraphs['paragraphs']):
#         contexts = paragraph['context']
#         qas_list = paragraph['qas']
#         for qas in tqdm(qas_list):
#             question = qas['question']
#             id_ = qas['id']
#             answ = qas['answers']
#             if len(answ) > 1:
#                 print('more than answers!')
#             if len(answ) == 0:
#                 continue
#             text = answ[0]['text']
#             answ_start = answ[0]['answer_start']
#             answ_end = answ_start + len(text)
#             is_impossible = qas['is_impossible']
#             data_prepare.append([id_, contexts, question, answ_start, answ_end, text, is_impossible])

# len(data_prepare)
# pickle.dump(data_prepare, open('./data_raw/data_prepare.pkl', 'wb'))


# [id_, contexts, question, answ_start, answ_end, text, is_impossible]
data_prepare = pickle.load(open('./data_raw/data_prepare.pkl', 'rb'))

# Q&A model
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim


cache_dir = 'D:/NLP/tokenizer_cache'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', cache_dir=cache_dir)

question, text = data_prepare[0][2],data_prepare[0][1]
inputs = tokenizer(question, text, return_tensors='pt')


start_positions = torch.tensor([data_prepare[0][3]])
end_positions = torch.tensor([data_prepare[0][4]])


outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
# scores 包含了 [CLS] question [SEP] passage 每个词汇的得分  
# start_scores 和 end_scores 得分最高的都在 [CLS]上意味着 机器学的说这题没有答案
start_scores = outputs.start_logits
len(start_scores[0]), len(inputs['input_ids'][0]) 

end_scores = outputs.end_logits
start_scores.argmax(), end_scores.argmax()

len(inputs['input_ids'][0])
len(start_scores)

# set seed
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 23333
set_seed(seed)

# train test spilt
def train_test_spilt(data, test_size=0.4):
    random.shuffle(data)
    data_trn = data[:int(len(data)*(1-0.4))]
    data_val = data[int(len(data)*(1-0.4)):]
    return data_trn, data_val

data_trn, data_val = train_test_spilt(data_prepare, test_size=0.4)



# Tokenizer
def get_answer_position(token_ids, answer_text, passages):
    token_id, text = token_ids, passages

    answer = answer_text
    answer_id = tokenizer(answer, return_tensors='pt')['input_ids'][0][1:-1]

    start_positions = 0
    end_positions = 0
    flag = 0
    for start in np.where(token_id==answer_id[0])[0]:
        i = 0
        if len(answer_id) == 1:
            start_positions = start
            end_positions = start+i
            break
        while(True):
            i += 1
            if i == len(answer_id)-1:
                if token_id[start+i] == answer_id[i]:
                    start_positions = start
                    end_positions = start+i
                    flag = 1   # 找到正确答案退出所有循环
                    break
            if token_id[start+i] != answer_id[i]:
                break
        if flag == 1:
            break
    
    return  start_positions, end_positions

class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True, model_name='bert-base-uncased'):
        self.data = data  # pandas dataframe

        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        # 根据自己输入data的格式修改
        context = self.data[index][1]
        question = self.data[index][2]
        answer_text = self.data[index][5]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(question, context, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            # print(token_ids.shape)
            try:
                start_positions, end_positions = get_answer_position(token_ids, answer_text, context)
            except:
                print(context, '\n', answer_text)
            start_positions = torch.tensor(start_positions)
            end_positions = torch.tensor(end_positions)
            return token_ids, attn_masks, token_type_ids, start_positions, end_positions
        else:
            return token_ids, attn_masks, token_type_ids

batch_size = 16
dataset_trn = CustomDataset(data_trn, 128)
loader_trn = Data.DataLoader(dataset_trn, batch_size, False)

dataset_val = CustomDataset(data_val, 128)
loader_val = Data.DataLoader(dataset_val, batch_size, False)


# token_ids, attn_masks, token_type_ids, ans_start, ans_end = next(iter(loader_trn))
# token_ids.shape, ans_start.shape
# tokenizer.convert_ids_to_tokens(token_ids[0][ans_start[0]: ans_end[0]+1])


class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name='bert-base-uncased', hidden_size=768):
        super(MyModel, self).__init__()
        self.bert = BertForQuestionAnswering.from_pretrained(model_name, 
                    output_hidden_states=True, return_dict=True, 
                    cache_dir=cache_dir)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        
    def forward(self, input_id, attn_masks, token_type):
        # bert_out (last_hidden_states, pool, all_hidden_states)
        # last_hidden_states [bs, seq_len, hid_size=768]
        # pool [bs, hid_size=768]
        # all_hidden_states (embedding, 各层的hidden_states, ....)
        output = self.bert(input_ids=input_id, token_type_ids=token_type, attention_mask=attn_masks) 
        return output.start_logits, output.end_logits


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },'./model_bert.pth')
    print('The best model has been saved')

def train_eval(model, optimizer, creterion, loader_trn, loader_val, epochs=2, continue_=True):
    if continue_:
        try:
            checkpoint = torch.load('./model_bert.pth', map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('-----Continue Training-----')
        except:
            print('No Pretrained model!')
            print('-----Training-----')
    else:
        print('-----Training-----')
    
    model = model.to(device)
    loss_his = []
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for _, batch in enumerate(tqdm(loader_trn)):
            
            optimizer.zero_grad()
            
            batch = [x.to(device) for x in batch]
            ans_start, ans_end = model(batch[0], batch[1], batch[2])
            
            loss_start = creterion(ans_start, batch[3])
            loss_end = creterion(ans_end, batch[4])
            loss = loss_start + loss_end 
            
            loss_his.append(loss.item())

            loss.backward()
            optimizer.step()
        
        print(loss.item())
        if epoch % 2 == 0:
            eval(model, optimizer, loader_val)
    
    return loss_his

best_score = 0.0
def eval(model, optimizer, loader_val):
    model.eval()
    answer_pred, answer_real = [], []
    
    def get_answer_origional(x, as_, ae_):
        if as_ > ae_:
            as_, ae_ = 0, 0 
        return list(x[as_: ae_+1])


    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader_val)):

            batch = [x.to(device) for x in batch]
            ans_start, ans_end = model(batch[0], batch[1], batch[2])
            ans_start, ans_end = ans_start.argmax(dim=1), ans_end.argmax(dim=1)
            
            ans_start = ans_start.detach().cpu().numpy()
            ans_end = ans_end.detach().cpu().numpy()
            input_ids = batch[0].detach().cpu().numpy()

            pred = [get_answer_origional(x, as_, ae_) for x,as_,ae_ in zip(input_ids, ans_start, ans_end)]
            real = [get_answer_origional(x, as_, ae_) for x,as_,ae_ in zip(input_ids, batch[3], batch[4])]

            answer_pred = answer_pred + pred
            answer_real = answer_real + real

    scores = []
    for p, r in zip(answer_pred, answer_real):
        scores.append(1.0-(len(set(r).difference(set(p))) / len(r)))
    
    print("Validation scores: {}".format(np.mean(scores)))
    global best_score
    if best_score < np.mean(scores):
        best_score = np.mean(scores)
        save(model, optimizer)


model = MyModel()
batch = next(iter(loader_trn))

creterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)

ans_start, ans_end, loss = model(batch[0], batch[1], batch[2])
optimizer.zero_grad()
loss.backward()
optimizer.step()



# ===============================================================
# 不使用 BertForQuestionAnswering
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name='bert-base-uncased', hidden_size=768):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, 
                    output_hidden_states=True, return_dict=True, 
                    cache_dir=cache_dir)
        self.linear_start = nn.Parameter(torch.zeros(768))
        self.linear_end = nn.Parameter(torch.zeros(768))
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        
    def forward(self, input_id, attn_masks, token_type):
        # bert_out (last_hidden_states, pool, all_hidden_states)
        # last_hidden_states [bs, seq_len, hid_size=768]
        # pool [bs, hid_size=768]
        # all_hidden_states (embedding, 各层的hidden_states, ....)
        outputs = self.bert(input_ids=input_id, token_type_ids=token_type, attention_mask=attn_masks) 
        last_hidden_states = outputs.last_hidden_state # [bs, seq_len, hid_dim]
        bs = last_hidden_states.shape[0]
        
        start_logits = torch.zeros(bs, last_hidden_states.shape[1]).to(device)
        end_logits = torch.zeros(bs, last_hidden_states.shape[1]).to(device)

        for i in range(last_hidden_states.shape[1]):
            hid_vector = last_hidden_states[:, i, :]  # [bs, hid_dim]
            start_scores = torch.bmm(hid_vector.unsqueeze(1),  \
                self.linear_start.repeat(bs, 1).unsqueeze(2)).squeeze()
            end_scores = torch.bmm(hid_vector.unsqueeze(1),  \
                self.linear_end.repeat(bs, 1).unsqueeze(2)).squeeze()

            start_logits[:, i] = start_scores
            end_logits[:, i] = end_scores

        return start_logits, end_logits


creterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)

model = MyModel()
batch = next(iter(loader_trn))

ans_start, ans_end = model(batch[0], batch[1], batch[2])
ans_start.shape



loss_start = creterion(ans_start, batch[3])
loss_end = creterion(ans_end, batch[4])
loss = loss_start + loss_end 

optimizer.zero_grad()
loss.backward()
optimizer.step()

