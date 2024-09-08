from hyperparams import model_lm,tokenizer_gpt,GENERATION_CONFIG,MAX_LENGTH
from utils import seq2seq_metric,SaveDeleteStateCallback
from functools import partial
from process_data import TrainDataset,ValidationDataset
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer,DataCollatorForLanguageModeling
import pickle
import torch
import gc
import pandas as pd

with open("./data/train_ds.pickle","rb") as train_f:
    train_ds:TrainDataset=pickle.load(train_f)
    train_ds=train_ds
with open("./data/validation_ds.pickle","rb") as val_f:
    validation_ds:ValidationDataset=pickle.load(val_f)
    validation_ds=validation_ds


collator=DataCollatorForLanguageModeling(
    tokenizer=tokenizer_gpt,
    return_tensors="pt",
    mlm=False,
)


compute_metrics=partial(seq2seq_metric,tokenizer=tokenizer_gpt)
model_lm.train(mode=True)

args=Seq2SeqTrainingArguments(
    output_dir="./saved/models/gpt2",
    overwrite_output_dir=True,
    weight_decay=.005,
    learning_rate=5e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=40,
    label_smoothing_factor=.3,
    warmup_ratio=.1,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_first_step=True,
    load_best_model_at_end=True,
    log_level='critical',
    metric_for_best_model="bleu",
    predict_with_generate=True,
    generation_config=GENERATION_CONFIG,
    save_total_limit=2,
    logging_steps=187,
    seed=12345,
    max_grad_norm=10.0,
    fp16=True,
)
trainer=Seq2SeqTrainer(
    model=model_lm,
    args=args,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=validation_ds,
    tokenizer=tokenizer_gpt,
    compute_metrics=compute_metrics,
    callbacks=[SaveDeleteStateCallback(model_lm)],
)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
trainer.train()
hist=trainer.state.log_history
df=pd.DataFrame(hist)
df.to_csv("./saved/history/history.csv",sep=",")