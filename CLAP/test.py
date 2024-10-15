import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, Audio
from transformers import ClapAudioModel, ClapFeatureExtractor, ClapAudioModelWithProjection

###
# hardware status
###

batch_size = 16
model_checkpoint = "laion/clap-htsat-unfused"

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

feature_extractor = ClapFeatureExtractor.from_pretrained(model_checkpoint)
def preprocess_function(examples):
  audio_array = [x["array"] for x in examples['filepath_deg']]
  inputs = feature_extractor(
    audio_array,
    sampling_rate=48000,
    return_tensors = 'pt'
  )
  
  labels = torch.tensor(examples['mos']).unsqueeze(-1).float()
  return {
    'input_values': inputs['input_features'],
    'label': labels
  }

###
# loading & preprocessing 5 testsets
###

print('\n*** NISQA VAL SIM ***')
NISQA_VAL_SIM = load_dataset(path=parent_dir+"/NISQA_VAL_SIM", data_files='NISQA_VAL_SIM_file.csv')['train']
NISQA_VAL_SIM = NISQA_VAL_SIM.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=48000))
NISQA_VAL_SIM = NISQA_VAL_SIM.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])
print(NISQA_VAL_SIM)

print('\n*** NISQA VAL LIVE ***')
NISQA_VAL_LIVE = load_dataset(path=parent_dir+"/NISQA_VAL_LIVE", data_files='NISQA_VAL_LIVE_file.csv')['train']
NISQA_VAL_LIVE = NISQA_VAL_LIVE.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=48000))
NISQA_VAL_LIVE = NISQA_VAL_LIVE.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])
print(NISQA_VAL_LIVE)

print('\n*** NISQA TEST FOR ***')
NISQA_TEST_FOR = load_dataset(path=parent_dir+"/NISQA_TEST_FOR", data_files='NISQA_TEST_FOR_file.csv')['train']
NISQA_TEST_FOR = NISQA_TEST_FOR.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=48000))
NISQA_TEST_FOR = NISQA_TEST_FOR.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])
print(NISQA_TEST_FOR)

print('\n*** NISQA TEST P501 ***')
NISQA_TEST_P501 = load_dataset(path=parent_dir+"/NISQA_TEST_P501", data_files='NISQA_TEST_P501_file.csv')['train']
NISQA_TEST_P501 = NISQA_TEST_P501.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=48000))
NISQA_TEST_P501 = NISQA_TEST_P501.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])
print(NISQA_TEST_P501)

print('\n*** NISQA TEST LIVETALK ***')
NISQA_TEST_LIVETALK = load_dataset(path=parent_dir+"/NISQA_TEST_LIVETALK", data_files='NISQA_TEST_LIVETALK_file.csv')['train']
NISQA_TEST_LIVETALK = NISQA_TEST_LIVETALK.map(update_filepath).cast_column("filepath_deg", Audio(sampling_rate=48000))
NISQA_TEST_LIVETALK = NISQA_TEST_LIVETALK.map(preprocess_function, batched=True).select_columns(['input_values', 'label'])
print(NISQA_TEST_LIVETALK)

###
# model loading from checkpoint 2249
###

foundation_model = ClapAudioModelWithProjection.from_pretrained(model_checkpoint)

class CustomModel(torch.nn.Module):
  def __init__(self, base_model):
    super(CustomModel, self).__init__()
    self.base_model = base_model
    self.mos_head = torch.nn.Linear(base_model.config.projection_dim, 1)  # Adjust hidden_size as per your model

  def forward(self, input_values, labels=None):
    base_model_output = self.base_model(input_values)
    audio_embeddings = base_model_output.audio_embeds
    return self.mos_head(audio_embeddings)

model = CustomModel(foundation_model).to(device)
model.load_state_dict(torch.load("clap-htsat-unfused-finetuned/checkpoint-2076/pytorch_model.bin", map_location=torch.device(device)))

print(f'CLAP model with MOS prediction head:\n {model}')

###
# arguments, compilation, & testing 
###

args = TrainingArguments(
  './testing',                            # checkpoints saved here
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
# model training and evaluation via rmse objective
###

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs
    
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(logits.squeeze(), labels.squeeze())
    return (loss, {"label": outputs}) if return_outputs else loss

trainer = CustomTrainer(
  model=model,
  args=args,
)

testsets = ['NISQA_VAL_LIVE', 'NISQA_TEST_FOR', 'NISQA_TEST_P501', 'NISQA_TEST_LIVETALK', 'NISQA_VAL_SIM']
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

    pred_mos = np.array(output[0]).flatten()
    true_mos = np.array(output[1]).flatten()
 
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