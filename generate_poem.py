from transformers import GenerationConfig,GPT2LMHeadModel
from utils import get_rhyming_words_ids,get_last_token
from functools import partial
import torch
torch.set_default_device("cuda")
from hyperparams import GENERATION_CONFIG,MAX_LENGTH,tokenizer_gpt,STOPPING_CRITERIA,PUNCTUATION_IDS,SINGLE_LETTERS,NUMBERS,pretrained
from typing import List,Literal
import dotenv
import os
dotenv.load_dotenv()

def generate_poem(starter:str,rhyme_scheme:str,mode:Literal["line_based","poem_based"]="line_based"):
    model_lm=GPT2LMHeadModel.from_pretrained(pretrained)
    not_allowed_start=[i[0] for i in [*PUNCTUATION_IDS,*SINGLE_LETTERS,*NUMBERS]]
    get_rhyme_token=partial(get_last_token,not_allowed_tokens=[*not_allowed_start,tokenizer_gpt.eos_token_id,tokenizer_gpt.pad_token_id])
    get_generation_config_for_rhyming_word=lambda :GenerationConfig(
        no_repeat_ngram_size=2,
        repetition_penalty=2.,
        top_k=80,
        top_p=0.95,
        max_new_tokens=1,
        temperature=1.5,
        num_beams=10,
        num_beam_groups=5,
        diversity_penalty=1.2,
        min_new_tokens=1,
        bad_words_ids=[*PUNCTUATION_IDS,*SINGLE_LETTERS,*NUMBERS,[tokenizer_gpt.eos_token_id],[tokenizer_gpt.pad_token_id],[512],[24564],[43817],[42061],[14329]],
        pad_token_id=tokenizer_gpt.pad_token_id,
        eos_token_id=tokenizer_gpt.eos_token_id,
    )
    poem:List[str]=[]
    poem.append(starter)
    rhyming_letter={}
    model=model_lm.eval()
    starter_tokens=tokenizer_gpt.encode(starter,
                                        padding="max_length",
                                        return_attention_mask=True,
                                        max_length=MAX_LENGTH,
                                        truncation=True,
                                        return_tensors="pt",)
    prev_line_length=starter_tokens.size(1)
    last_line_ids=starter_tokens
    last_word_for_rhyme=tokenizer_gpt.decode(get_rhyme_token(starter_tokens[0]))
    rhyming_letter[rhyme_scheme[0]]=get_rhyming_words_ids(last_word_for_rhyme)
    last_line=tokenizer_gpt.decode(last_line_ids[0])
    match mode:
        case "line_based":
            for letter in rhyme_scheme[1:]:
                prev_line_length=last_line_ids.size(1)
                last_line_attention_mask=(last_line_ids!=tokenizer_gpt.pad_token_id).long()
                last_line_ids=model.generate(last_line_ids,
                                         generation_config=GENERATION_CONFIG,
                                         stopping_criteria=STOPPING_CRITERIA,
                                         attention_mask=last_line_attention_mask)[::,prev_line_length:]
                last_line=tokenizer_gpt.decode(last_line_ids[0],skip_special_tokens=True)
                if rhyming_letter.get(letter) is None:
                    last_word_for_rhyme=get_rhyme_token(last_line_ids[0]).item()
                    last_word=tokenizer_gpt.decode([last_word_for_rhyme],skip_special_tokens=True)
                    rhyming_letter[letter]=get_rhyming_words_ids(last_word)
                else:
                    last_word=last_line_ids[-1]
                    length=last_line_ids.size(1)
                    gen_config=get_generation_config_for_rhyming_word()
                    prefix_allowed_tokens_fun=lambda batch_id,input_ids:rhyming_letter[letter]
                    rhyming=model.generate(last_line_ids,generation_config=gen_config,prefix_allowed_tokens_fn=prefix_allowed_tokens_fun)[::,length::]
                    rhyming_word=tokenizer_gpt.decode(rhyming[0])
                    print(f"rhyming word: {rhyming_word}")
                    last_line=last_line+rhyming_word
                    last_line_ids=torch.cat([last_line_ids,rhyming,torch.tensor([[tokenizer_gpt.eos_token_id]])],dim=-1)
                poem.append(last_line)
            for i in range(len(rhyme_scheme)):
                poem[i]=poem[i].replace('\n','').strip()
            poem="\n".join(poem).replace("<|endoftext|>","").replace("<|pad|>","")
            return poem
        case "poem_based":
            poem_tokens=starter_tokens[0]
            cumsum=[0,poem_tokens.size(-1)]
            for letter in rhyme_scheme[1:]:
                if len(poem_tokens.shape)!=2:
                    poem_tokens=poem_tokens.unsqueeze(0)
                poem_attention_mask=(poem_tokens!=tokenizer_gpt.pad_token_id).long().expand_as(poem_tokens)
                poem_tokens=model.generate(poem_tokens,
                                         generation_config=GENERATION_CONFIG,
                                         stopping_criteria=STOPPING_CRITERIA,
                                         attention_mask=poem_attention_mask)
                last_line=tokenizer_gpt.decode(last_line_ids[0],skip_special_tokens=True)
                if rhyming_letter.get(letter) is None:
                    last_word_for_rhyme=get_rhyme_token(poem_tokens[0]).item()
                    last_word=tokenizer_gpt.decode([last_word_for_rhyme],skip_special_tokens=True)
                    rhyming_letter[letter]=get_rhyming_words_ids(last_word)
                    cumsum.append(poem_tokens.size(-1))
                else:
                    length=poem_tokens.size(1)
                    gen_config=get_generation_config_for_rhyming_word()
                    prefix_allowed_tokens_fun=lambda batch_id,input_ids:rhyming_letter[letter]
                    rhyming=model.generate(poem_tokens,generation_config=gen_config,prefix_allowed_tokens_fn=prefix_allowed_tokens_fun)[::,length]
                    rhyming_word=tokenizer_gpt.decode(rhyming[0])
                    poem_tokens=torch.cat([poem_tokens,rhyming.unsqueeze(0)],dim=-1)
                    cumsum.append(poem_tokens.size(-1))
                    print(f"rhyming word: {rhyming_word}")
                    last_line=last_line+" "+rhyming_word
            tokens_parted=[]
            for i in range(1,len(cumsum)):
                line_range=slice(cumsum[i-1],cumsum[i])
                line=poem_tokens[0,line_range]
                decoded=tokenizer_gpt.decode(line,skip_special_tokens=True)
                tokens_parted.append(decoded)
            return "\n".join(tokens_parted)
        case _:
            raise NotImplementedError("Sadece iki şiir oluşturma metodu mevcuttur.")
if __name__=="__main__":
    poem=generate_poem(os.getenv("TEST_STARTER_LINE"),"ABAB",mode="poem_based")
    print(poem)