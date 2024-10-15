import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, Audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

###
# hardware status
###

batch_size = 16
model_checkpoint = "superb/hubert-base-superb-ks"

print('Availability of torch: ', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for model training & evaluation: ', device)

###
# data preprocessing functions
###

parent_dir = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus'
def update_filepath(example):
  example['filepath_deg'] = os.path.join(parent_dir, example['filepath_deg'])
  return example

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
def preprocess_function(examples):
  audio_array = [x["array"] for x in examples['filepath_deg']]
  inputs = feature_extractor(
    audio_array,
    truncation=True,
    sampling_rate=16000,
    max_length=16000*10,
    padding='max_length',
    return_tensors = 'pt'
  )
  
  labels = torch.tensor(examples['mos']).unsqueeze(-1).float()
  return {
    'input_values': inputs['input_values'],
    'label': labels
  }

###
# loading & preprocessing 5 testsets
###

print('\n*** NISQA VAL SIM ***')
NISQA_VAL_SIM = load_dataset(path=parent_dir+"/NISQA_VAL_SIM", data_files='NISQA_VAL_SIM_file.csv')['train']
NISQA_VAL_SIM = NISQA_VAL_SIM.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=16000))
NISQA_VAL_SIM = NISQA_VAL_SIM.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])

print('\n*** NISQA VAL LIVE ***')
NISQA_VAL_LIVE = load_dataset(path=parent_dir+"/NISQA_VAL_LIVE", data_files='NISQA_VAL_LIVE_file.csv')['train']
NISQA_VAL_LIVE = NISQA_VAL_LIVE.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=16000))
NISQA_VAL_LIVE = NISQA_VAL_LIVE.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])

print('\n*** NISQA TEST FOR ***')
NISQA_TEST_FOR = load_dataset(path=parent_dir+"/NISQA_TEST_FOR", data_files='NISQA_TEST_FOR_file.csv')['train']
NISQA_TEST_FOR = NISQA_TEST_FOR.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=16000))
NISQA_TEST_FOR = NISQA_TEST_FOR.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])

print('\n*** NISQA TEST P501 ***')
NISQA_TEST_P501 = load_dataset(path=parent_dir+"/NISQA_TEST_P501", data_files='NISQA_TEST_P501_file.csv')['train']
NISQA_TEST_P501 = NISQA_TEST_P501.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=16000))
NISQA_TEST_P501 = NISQA_TEST_P501.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])

print('\n*** NISQA TEST LIVETALK ***')
NISQA_TEST_LIVETALK = load_dataset(path=parent_dir+"/NISQA_TEST_LIVETALK", data_files='NISQA_TEST_LIVETALK_file.csv')['train']
NISQA_TEST_LIVETALK = NISQA_TEST_LIVETALK.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=16000))
NISQA_TEST_LIVETALK = NISQA_TEST_LIVETALK.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])

###
# model loading from checkpoint 2076
###

model = HubertForSequenceClassification.from_pretrained(model_checkpoint).to(device)

hidden_size = model.classifier.in_features
num_output_classes = 1
model.classifier = torch.nn.Linear(hidden_size, num_output_classes)

model.load_state_dict(torch.load("hubert-base-superb-ks-finetuned/checkpoint-2076/pytorch_model.bin", map_location=torch.device(device)))
print(model)

###
# arguments, compilation, & testing 
###

args = TrainingArguments(
  './testing',                            # checkpoints saved here
  per_device_eval_batch_size=batch_size,
  dataloader_num_workers = 8,             # parallelizing data fetch process
  eval_accumulation_steps = 16,
)

###
# model evaluation via mse, pcc & srcc
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
)

testsets = ['NISQA_VAL_SIM', 'NISQA_VAL_LIVE', 'NISQA_TEST_FOR', 'NISQA_TEST_P501', 'NISQA_TEST_LIVETALK']
for testset in testsets:
    print('\nTestset under consideration: ', testset)

    if testset == 'NISQA_VAL_SIM':
      ds = NISQA_VAL_SIM
    elif testset == 'NISQA_VAL_LIVE':
        ds = NISQA_VAL_LIVE
    elif testset == 'NISQA_TEST_FOR':
        ds = NISQA_TEST_FOR
    elif testset == 'NISQA_TEST_P501':
        ds = NISQA_TEST_P501
    elif testset == 'NISQA_TEST_LIVETALK':
        ds = NISQA_TEST_LIVETALK
        
    output = trainer.predict(ds)
    
    pred_mos = np.array(output.predictions[0]).flatten()
    true_mos = np.array(output.label_ids).flatten()
    
    # plot & numerical analysis
    rmse = (np.mean((true_mos-pred_mos)**2))**0.5
    print('RMSE (utterance-level) = ', rmse)
    p_corr, _ = pearsonr(true_mos, pred_mos)
    print('Pearson Correlation Coefficient = ', p_corr)
    s_corr, _ = spearmanr(true_mos, pred_mos)
    print('Spearman Rank Correlation Coefficient = ', s_corr)

    # SCATTER PLOT
    plt.figure()
    plt.scatter(pred_mos, true_mos)
    plt.title(testset)
    plt.xlim([1,5]); plt.xlabel('Predicted')
    plt.ylim([1,5]); plt.ylabel('True MOS')
    plt.savefig('testing/' + testset + '.png')
  