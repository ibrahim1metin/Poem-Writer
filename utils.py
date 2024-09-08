from torch import FloatTensor, LongTensor
from typing import Optional
from transformers import StoppingCriteria,GPT2TokenizerFast,GPT2LMHeadModel
from transformers import TrainerCallback, PreTrainedModel
from transformers.trainer_callback import TrainerControl, TrainerState
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import accuracy as accuracy_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from transformers.training_args import TrainingArguments
import os
import json
from collections import Counter
from functools import reduce
from typing import List

class StopByTokenCriteria(StoppingCriteria):
    def __init__(self,eos_token_id=0,encounter_ratio=.5,min_length:Optional[int]=2) -> None:
        super().__init__()
        self.eos_token_id=eos_token_id
        self.min_length=min_length
        self.encounter_ratio=encounter_ratio
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        B_SIZE=input_ids.size(0)
        last_tokens=input_ids[::,-1]
        eos_count=(last_tokens==self.eos_token_id).sum().item()
        ratio=eos_count/B_SIZE
        is_done=ratio>self.encounter_ratio
        if is_done and self.min_length!=None and input_ids.shape[-1]<self.min_length:
            return False
        return is_done

class SaveDeleteStateCallback(TrainerCallback):
    def __init__(self,model:PreTrainedModel,state_dir="./saved/States",num_states=3,**kwargs) -> None:
        super().__init__()
        self.model=model
        self.state_dir=state_dir
        self.num_states=num_states
        os.makedirs(self.state_dir,exist_ok=True)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.model.training==False:
            return
        global_step=state.global_step
        curr_states=sorted(os.listdir(self.state_dir), key=lambda x: int(x.split('-')[-1]))
        if len(curr_states)>=self.num_states:
            first_file=curr_states[0]
            os.remove(
                os.path.join(self.state_dir,first_file)
            )
        self.model.save_pretrained(save_directory=os.path.join(self.state_dir,f"checkpoint-{global_step}"))



rouge_score=ROUGEScore()
bleu_score=BLEUScore()
def seq2seq_metric(p,tokenizer:GPT2TokenizerFast):
    predicted,true_labels=p
    if isinstance(predicted,np.ndarray):
        predicted=torch.tensor(predicted)
    if isinstance(true_labels,np.ndarray):
        true_labels=torch.tensor(true_labels)
    if predicted.size(1)!=true_labels.size(1):
        fused=torch.nn.utils.rnn.pad_sequence([predicted.T,true_labels.T],
                                              batch_first=True,
                                              padding_value=tokenizer.pad_token_id,)
        predicted=fused[0,::,::].T
        true_labels=fused[1,::,::].T
    
    metrics={}
    predicted=torch.where(predicted!=-100,predicted,tokenizer.pad_token_id)
    true_labels=torch.where(true_labels!=-100,true_labels,tokenizer.pad_token_id)
    assert torch.all((predicted >= 0) & (predicted <= tokenizer.vocab_size)), "Invalid token ids in predicted"
    assert torch.all((true_labels >= 0) & (true_labels <= tokenizer.vocab_size)), "Invalid token ids in true_labels"
    decoded_predicted=tokenizer.batch_decode(predicted,skip_special_tokens=True)
    decoded_true=tokenizer.batch_decode(true_labels,skip_special_tokens=True)
    accuracy=accuracy_score(predicted,true_labels,task="multiclass",num_classes=tokenizer.vocab_size*2)
    rouge=rouge_score(decoded_predicted,decoded_true)
    bleu=bleu_score(decoded_predicted,[decoded_true])
    metrics["accuracy"]=accuracy
    metrics["bleu"]=bleu
    metrics={**metrics,**rouge}
    return metrics

def create_inverse_vocab(vocab_file:str="./data/vocab.json"):
    with open(vocab_file,"r",encoding="utf-8",errors="ignore") as vocab_f:
        json_vocab=json.load(vocab_f)
    keys=sorted(list(json_vocab.keys()),key=lambda x: (x[-1] if x else '\t'))
    with open("./data/inverse_vocab.txt",'w',encoding="utf-8",errors="ignore") as keys_file:
        for key in keys:
            if len(key)>1:
                keys_file.write(f'{key}#{json_vocab[key]}\n')


iso2turkish={
    "Äħ":"ı",
    "Ä±":"ı",
    "Ã¼":"ü",
    "Ã¶":"ö",
    "Ã§":"ç",
    "ĊL":"ş",
    "ÄL":"ğ",
    "Äħ":"ı",
    "ĊL":"ş",
    "ÄL":"ğ",
    "GÄ°":"İ",
}

def get_cumsum():
    letters = []
    with open("./data/inverse_vocab.txt", 'r', encoding="utf-8", errors="ignore") as keys_file:
        lines = keys_file.readlines()
        for i in range(len(lines)):
            lines[i]=lines[i].split("#")[0]
        for i in range(len(lines)):
            for k, v in iso2turkish.items():
                lines[i] = lines[i].replace(k, v)
            if len(lines[i])>1 and lines[i][-1]=="Ä":
                lines[i]=lines[i][:-2]+lines[-1]
    for line in lines:
        if len(line) > 1:
            letters.append(line[-1])
        else:
            letters.append("<|pad|>")
    counts = Counter(letters)
    letters_u = counts.keys()
    counts_u = counts.values()
    cumsum = reduce(lambda a, x: a + [a[-1] + x] if a else [x], counts_u, [])
    cumsum_dict = {k: v for k, v in sorted(zip(letters_u, cumsum), key=lambda x: x[-1])}
    return cumsum_dict
CUMSUM=get_cumsum()
def get_not_rhyming_words(starting_word):
    indexes = CUMSUM.copy()
    letter = starting_word[-1]
    start_idx = list(indexes.values())[list(indexes.keys()).index(letter) - 1] if letter in indexes else 0
    end_idx = indexes.get(letter, len(indexes))
    with open("./data/inverse_vocab.txt", 'r', encoding="utf-8", errors="ignore") as vocab_f:
        lines = vocab_f.readlines()
    lines_out = lines[:start_idx] + lines[end_idx:]
    return lines_out,lines[start_idx:end_idx]
def get_bad_words_ids(starting_word):
    bad_words,_=get_not_rhyming_words(starting_word)
    bad_words=[bad_word[:-1] if bad_word.endswith("\n") else bad_word for bad_word in bad_words]
    ids=[]
    for word in bad_words:
        ids.append(
            [int(word.split("#")[-1])]
        )
    return ids

def get_last_token(tokens,not_allowed_tokens):
    i=tokens.size(-1)-1
    while i>=0:
        if not (tokens[i] in not_allowed_tokens):
            return tokens[i]
        i-=1

def get_rhyming_words_ids(starting_word):
    _,rhyming_words=get_not_rhyming_words(starting_word)
    bad_words=[rhymin_word[:-1] if rhymin_word.endswith("\n") else rhymin_word for rhymin_word in rhyming_words]
    ids=[]
    for word in bad_words:
        ids.append(
            [int(word.split("#")[-1])]
        )
    return ids


def find_numbers():
    start=200
    end=865
    with open("./data/inverse_vocab.txt", 'r', encoding="utf-8", errors="ignore") as vocab_f:
        lines=vocab_f.readlines()
    numbers=lines[start:end]
    number_ids=[[int(number.split("#")[-1][:-1])] for number in numbers]
    return number_ids
def find_punctuation():
    start=865
    end=895
    with open("./data/inverse_vocab.txt", 'r', encoding="utf-8", errors="ignore") as vocab_f:
        lines=vocab_f.readlines()
    numbers=lines[start:end]
    p_ids=[[int(number.split("#")[-1][:-1])] for number in numbers]
    return p_ids
if __name__=="__main__":
    words,r=(get_not_rhyming_words("r"))
    words=[word.split("#")[0] for word in words]
    for word in words:
        if word.endswith("r"):
            print(word)
    ids=get_bad_words_ids("e")
    rhy=get_rhyming_words_ids("r")
    print(r)
    print(rhy)
