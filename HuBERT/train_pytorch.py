'Adapting HuBERT from Hugging Face for Speech Quality Assessment'

import os
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, DatasetDict, Audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

print('Availability of torch: ', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for model training & evaluation: ', device)

batch_size = 16
model_checkpoint = "superb/hubert-base-superb-ks"

###
# load audio dataset from csv files
###

df_NISQA_TRAIN_SIM = pd.read_csv('/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv')
df_NISQA_TRAIN_LIVE = pd.read_csv('/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_file.csv')

train_df = pd.concat([df_NISQA_TRAIN_SIM, df_NISQA_TRAIN_LIVE], axis=0, ignore_index=True)
train_df.to_csv('temp/NISQA_TRAIN.csv', index=False)

df_NISQA_VAL_SIM = pd.read_csv('/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
df_NISQA_VAL_LIVE = pd.read_csv('/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')

val_df = pd.concat([df_NISQA_VAL_SIM, df_NISQA_VAL_LIVE], axis=0, ignore_index=True)
val_df.to_csv('temp/NISQA_VAL.csv', index=False)

###
# convert pandas dataframe to datasets format
###

train_ds = load_dataset(path="temp", data_files='NISQA_TRAIN.csv')['train']
val_ds = load_dataset(path="temp", data_files='NISQA_VAL.csv')['train']

combined_dataset = DatasetDict({
  'train': train_ds,
  'validation': val_ds,
}).select_columns(['filepath_deg', 'mos'])
print(f'\ncsv dataset:\n {combined_dataset}')

###
# sample degraded audio from its filepath
###

parent_dir = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus'
def update_filepath(example):
  example['filepath_deg'] = os.path.join(parent_dir, example['filepath_deg'])
  return example

combined_dataset = combined_dataset.map(update_filepath)
audio_dataset = combined_dataset.cast_column("filepath_deg", Audio(sampling_rate=16000))
print(f'\naudio dataset:\n {audio_dataset}')

###
# compute features (compatible wav2vec input) from raw audio.
###

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
def preprocess_function(examples):
  audio_array = [x["array"] for x in examples['filepath_deg']]
  inputs = feature_extractor(
    audio_array,
    truncation=True,
    return_tensors='pt',
    max_length=16000*10,    
    sampling_rate=16000,
    padding='max_length',
  )
  
  labels = torch.tensor(examples['mos']).unsqueeze(-1).float()
  return {
    'input_values': inputs['input_values'],
    'label': labels
  }
  
encoded_dataset = audio_dataset.map(preprocess_function, batched=True).select_columns(['input_values', 'label']).shuffle()
print(f'\nencoded dataset:\n {encoded_dataset}')

###
# load pretrained wav2vec with 1-unit head to predict mos
###

model = HubertForSequenceClassification.from_pretrained(model_checkpoint).to(device)

hidden_size = model.classifier.in_features
num_output_classes = 1
model.classifier = torch.nn.Linear(hidden_size, num_output_classes)

print(f'HuBERT model with MOS prediction head:\n {model}')

###
# model optimization and checkpointing
###

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
  f"{model_name}-finetuned",              # checkpoints saved here
  evaluation_strategy = "epoch",          # evaluation at the end of epoch
  save_strategy = "epoch",                # model saved at the end of epoch
  learning_rate=5e-5,                     # value obtained from AST paper
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  num_train_epochs=20,                    
  logging_strategy = "epoch",             # results logged at the end of epoch
  dataloader_num_workers = 8,             # parallelizing data fetch process
  load_best_model_at_end=True,            # best model loaded after training
  greater_is_better="False"               # specifying lower mse is better
)

###
# model training and evaluation via mse objective
###

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(logits.squeeze(), labels.squeeze())
    return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
  model=model,
  args=args,
  train_dataset=encoded_dataset['train'],
  eval_dataset=encoded_dataset['validation'],
)

trainer.train()
