import pickle
from torch.utils.data import Dataset
import torch
from hyperparams import tokenizer_gpt,MAX_LENGTH,TRAIN_RATIO,DEVICE
import os
import dotenv
dotenv.load_dotenv()

class TrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.inps=encoded_inputs_train
    def __len__(self):
        return self.inps.input_ids.size(0)
    def __getitem__(self, idx):
        sample={
            "input_ids":self.inps.input_ids[idx],
            "attention_mask":self.inps.attention_mask[idx],
            "labels":self.inps.input_ids[idx]
        }
        return sample
class ValidationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.inps=encoded_inputs_validation
    def __len__(self):
        return self.inps.input_ids.size(0)
    def __getitem__(self, idx):
        sample={
            "input_ids":self.inps.input_ids[idx],
            "attention_mask":self.inps.attention_mask[idx],
            "labels":self.inps.input_ids[idx]
        }
        return sample
if __name__=="__main__":
    with open(os.getenv("DATASET_1"),mode="r",encoding="utf-8") as data_f:
        total_f=data_f.readlines()
    inputs=total_f
    outputs=total_f


    trainLength=int(len(inputs)*TRAIN_RATIO)

    inputs_tr=inputs[:trainLength]
    val_inputs=inputs[trainLength:]

    outputs_tr=outputs[:trainLength]
    val_outputs=outputs[trainLength:]

    encoded_inputs_train=tokenizer_gpt.batch_encode_plus(
        inputs_tr,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    encoded_inputs_train["input_ids"]=encoded_inputs_train["input_ids"].to(DEVICE)
    encoded_inputs_train["attention_mask"]=encoded_inputs_train["attention_mask"].to(DEVICE)

    encoded_inputs_validation=tokenizer_gpt.batch_encode_plus(
        val_inputs,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    encoded_inputs_validation["input_ids"]=encoded_inputs_validation["input_ids"].to(DEVICE)
    encoded_inputs_validation["attention_mask"]=encoded_inputs_validation["attention_mask"].to(DEVICE)
    with open("./data/train_ds.pickle","wb") as train_f:
        pickle.dump(TrainDataset(),train_f)
    with open("./data/validation_ds.pickle","wb") as val_f:
        pickle.dump(ValidationDataset(),val_f)
    print(encoded_inputs_train.input_ids.shape)
    train_ds=TrainDataset()
    print(train_ds[128])
    print(encoded_inputs_train.input_ids.device)