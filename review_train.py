from custom_models import MultimodalLMV3ForTokenClass
import argparse
import json
import sys
import ast
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_scheduler
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# since only supporting BIO type models can just set these as constant, no need to infer
NULL_TAG = "O"
TAG_TYPES = ["O", "B-", "I-"]
ID2TYPE = {i:t for i, t in enumerate(TAG_TYPES)}
TYPES2ID = {t:i for i, t in enumerate(TAG_TYPES)}

# training hyperparameters
NULL_PERCENT = 7
NUM_EPOCHS = 5


def init_params():
  parser = argparse.ArgumentParser(prog="review_train.py", description="review and/or train a given model on given data")
  parser.add_argument("model_dir")
  parser.add_argument("datafile")
  parser.add_argument("label_types")
  parser.add_argument("batch_size")


def normalize_bbox(bbox, width, height) -> tuple:
  return (
    int(1000*(bbox[0]/width)),
    int(1000*(bbox[1]/height)),
    int(1000*(bbox[2]/width)),
    int(1000*(bbox[3]/height))
  )


def unnormalize_box(bbox, width, height):
  return [
    width * (bbox[0] / 1000),
    height * (bbox[1] / 1000),
    width * (bbox[2] / 1000),
    height * (bbox[3] / 1000),
  ]


def encode_review_dataset(example, processor):
  image = Image.open(example["image_path"]).convert("RGB")
  words = example["words"]
  bboxs = [normalize_bbox(bbox, example["page_width"], example["page_height"]) for bbox in example["bboxs"]]

  with torch.no_grad():
    encoded_inputs = processor(
      image, words, boxes=bboxs, padding="max_length",
      truncation=True, return_tensors="pt"
    ).to(DEVICE)
  
  return encoded_inputs


def collate_review(batch):
  input_ids = [torch.tensor(example["input_ids"]) for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
  bbox = [torch.tensor(example["bbox"]) for example in batch]
  pixel_values = [torch.tensor(example["pixel_values"]) for example in batch]

  input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
  if len(input_ids_padded.shape) > 2:
    input_ids_padded = input_ids_padded.squeeze()
  if len(input_ids_padded.shape) == 1:
    input_ids_padded = input_ids_padded.unsqueeze(-1)
  attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  if len(attention_mask_padded.shape) > 2:
    attention_mask_padded = attention_mask_padded.squeeze()
  if len(attention_mask_padded) == 1:
    attention_mask_padded = attention_mask_padded.unsqueeze(-1)
  bbox_padded = pad_sequence(bbox, batch_first=True, padding_value=0)
  if len(bbox_padded.shape) > 2:
    bbox_padded = bbox_padded.squeeze()
  if len(bbox_padded) == 1:
    bbox_padded = bbox_padded.unsqueeze(-1)
  pixel_values_padded = pad_sequence(pixel_values, batch_first=True, padding_value=0)
  if len(pixel_values_padded.shape) > 2:
    pixel_values_padded = pixel_values_padded.squeeze()
  if len(pixel_values_padded) == 1:
    pixel_values_padded = pixel_values_padded.unsqueeze(-1)
  
  return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "bbox":bbox_padded, "pixel_values":pixel_values_padded, }


def data_iter(dataset, batch_size):
  for i in range(0, len(dataset), batch_size):
    yield dataset[i:i+batch_size]


# take a list of token predictions and word ids and realign so that predictions
# are at word level not token level
def align_token_to_word(predictions, word_ids):
  word_aligned_predictions = []
  current_id = -1
  for pred, id in zip(predictions, word_ids):
    if id is None:
      continue
    if current_id != id:
      word_aligned_predictions.append(pred)
      current_id = id
  return word_aligned_predictions


""" --------------------------------------------------------------
" REVIEW METHOD: ....
" ------------------------------------------------------------ """
def review_data(data, batch_size, model, processor):
  dataset = Dataset.from_list(data)
  encoded_dataset = dataset.map(encode_review_dataset, fn_kwargs={"processor":processor}, batched=False)
  eval_dataloader = DataLoader(encoded_dataset, batch_size, shuffle=False, collate_fn=collate_review)
  eval_data_iter = data_iter(encoded_dataset, batch_size)
  
  for i, (batch_collate, batch_data) in enumerate(zip(eval_dataloader, eval_data_iter)):
    batch_collate = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch_collate.items()} # ensure every tensor is on the same device as the model
    data[i*len(eval_dataloader)]["ner_tags"] = {}
    with torch.no_grad():
      for label_type in label_types:
        logits, loss = model(**batch_collate, label_typel=label_type)
        predictions = logits.argmax(-1).squeeze()
        for j, prediction in enumerate(predictions):
          prediction = align_token_to_word(prediction, batch_data[j].word_ids())
          true_prediction = [f"{ID2TYPE[p]}{label_type}" if ID2TYPE[p] != NULL_TAG else NULL_TAG for p in prediction]
          data[i*len(eval_dataloader)+j]["ner_tags"][label_type] = true_prediction
  
  return data # data now has a prediction field for each of the label types, this can be rewritten to the input file now



# alignes the list of word labels/predictions to there coresponding tokens
def align_word_to_token(text, boxes, word_labels, tokenizer):
  token_labels = []
  word_ids = tokenizer(text=text, boxes=boxes, padding='max_length', truncation=True).word_ids()
  current_word_idx = None
  #print(f"len words = {len(text)}", file=sys.stderr)
  #print(f"word labels: {word_labels}, {len(word_labels)}", file=sys.stderr)
  #print(f"word_ids: {word_ids}", file=sys.stderr)

  for _, word_idx in enumerate(word_ids):
    if word_idx is None:
      token_labels.append(-100) # -100 is automatically ignore in pytorch loss calculations
    elif word_idx != current_word_idx:
      current_word_idx = word_idx
      token_labels.append(word_labels[current_word_idx])
    else:
      token_labels.append(word_labels[current_word_idx])

  return token_labels[0:512]


# useful when converting list of token tags to there respective ids 
def convert_token_tags_to_ids(token_tags):
  token_ids = []
  for tag in token_tags:
    if tag != -100:
      token_ids.append(TYPES2ID[tag[:2]])
    else:
      token_ids.append(-100)
  return token_ids


def encode_train_dataset(example, label_types=None):
  image = Image.open(example["image_path"]).convert("RGB")
  words = example["words"]
  bboxs = [normalize_bbox(bbox, example["page_width"], example["page_height"]) for bbox in example["bboxs"]]

  with torch.no_grad():
    encoded_inputs = processor(
      image, words, boxes=bboxs, padding="max_length",
      truncation=True, return_tensors="pt"
    ).to(DEVICE)

  label_dict = example["ner_tags"]

  label_dict = {label_type:align_word_to_token(words, bboxs, label_dict[label_type], processor.tokenizer) for label_type in label_types}
  label_dict = {label_type:convert_token_tags_to_ids(label_dict[label_type]) for label_type in label_types}
  label_dict = {label_type:torch.tensor(label_dict[label_type], device=DEVICE, dtype=torch.long) for label_type in label_types}
  label_dict = {label_type:label_dict[label_type].to(DEVICE) for label_type in label_types}

  encoded_inputs["input_mask"] = example["input_mask"]
  encoded_inputs["labels_dict"] = label_dict
  return encoded_inputs


def create_input_mask(training_data, label_types):
  input_mask = [{label_type:1 for label_type in label_types} for example in training_data]
  null_cnt_dict = {label_type: 1 for label_type in label_types} # make a null counter for each label type

  for i, example in enumerate(training_data):
    
    ner_tags_dict = example["ner_tags"] # dict of all the kinds of tags
    for label_type in label_types:
      tags = ner_tags_dict[label_type]
      tags = [TYPES2ID[tag[:2]] for tag in tags] # convert tags to numbers
      null_cnt = null_cnt_dict[label_type]

      if max(tags) == 0 and null_cnt != 0: # if there are no others then 'O' tags this is a null example
        if null_cnt == NULL_PERCENT: # reset the null cnt
          null_cnt_dict[label_type] = 0
        else:
          null_cnt_dict[label_type] += 1
        input_mask[i][label_type] = 0 # therefore mask this example 
      else: # either this is not a null example or it there are enough non null examples to allow for a null example
        null_cnt_dict[label_type] += 1 # so just count and leave the mask untouched
  
  return input_mask


def generate_normalized_weights(dataset: list, label_types):
  weights = {label_type:[0 for _ in TAG_TYPES] for label_type in label_types}

  # find the frequencies of every tag in the dataset
  for example in dataset:
    for label_type in label_types:
      ner_tags = example["ner_tags"][label_type]
      if example["input_mask"][label_type] == 1: # only count examples that are masked
        for tag in ner_tags:
          weights[label_type][TYPES2ID[tag[:2]]] += 1
  
  # compute weights for loss as inverse of frequencies
  for label_type in label_types:
    tag_freqs = weights[label_type]
    total_tag_count = sum(tag_freqs)
    inv_freqs = [(total_tag_count / count) for count in tag_freqs]
    weights[label_type] = inv_freqs

  # Normalize the weights and convert to tensor format
  for label_type in label_types:
    inv_freqs = weights[label_type]
    min_weight = min(inv_freqs)
    normalized_weights = [(weight / min_weight) for weight in inv_freqs]
    weight_tensor = torch.tensor(normalized_weights, dtype=torch.float)
    weight_tensor = weight_tensor.to(DEVICE)
    weights[label_type] = weight_tensor
  
  return weights


def collate_train(batch):
  input_ids = [torch.tensor(example["input_ids"]) for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
  bbox = [torch.tensor(example["bbox"]) for example in batch]
  pixel_values = [torch.tensor(example["pixel_values"]) for example in batch]
  labels = {}
  input_mask = {}
  for label_type in label_types:
    labels[label_type] = [torch.tensor(example["labels_dict"][label_type]) for example in batch]
    labels[label_type] = pad_sequence(labels[label_type], batch_first=True, padding_value=-100)
    labels[label_type] = labels[label_type].to(DEVICE)
    input_mask[label_type] = [torch.tensor(example["input_mask"][label_type]) for example in batch] 
    input_mask[label_type] = torch.stack(input_mask[label_type]).unsqueeze(-1) # should just be shape (batch_size, 1) now
    input_mask[label_type] = input_mask[label_type].to(DEVICE)


  # pad
  input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
  if len(input_ids_padded.shape) > 2:
    input_ids_padded = input_ids_padded.squeeze()
  if len(input_ids_padded.shape) == 1:
    input_ids_padded = input_ids_padded.unsqueeze(-1)
  attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  if len(attention_mask_padded.shape) > 2:
    attention_mask_padded = attention_mask_padded.squeeze()
  if len(attention_mask_padded) == 1:
    attention_mask_padded = attention_mask_padded.unsqueeze(-1)
  bbox_padded = pad_sequence(bbox, batch_first=True, padding_value=0)
  if len(bbox_padded.shape) > 2:
    bbox_padded = bbox_padded.squeeze()
  if len(bbox_padded) == 1:
    bbox_padded = bbox_padded.unsqueeze(-1)
  pixel_values_padded = pad_sequence(pixel_values, batch_first=True, padding_value=0)
  if len(pixel_values_padded.shape) > 2:
    pixel_values_padded = pixel_values_padded.squeeze()
  if len(pixel_values_padded) == 1:
    pixel_values_padded = pixel_values_padded.unsqueeze(-1)

  return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "labels_dict": labels, "bbox":bbox_padded, "pixel_values":pixel_values_padded, "input_mask_dict":input_mask}



"""
NOTE: this must be changed if the model I end up selecting changes
"""
def train_data(data, batch_size, model, processor, model_dir):
  dataset = Dataset.from_list(data)
  input_mask = create_input_mask(train_data, label_types)
  dataset = dataset.add_column("input_mask", input_mask)
  weight_dict = generate_normalized_weights(dataset, label_types)
  encoded_dataset = dataset.map(encode_train_dataset, fn_kwargs={"processor":processor}, batched=False)

  train_dataloader = DataLoader(dataset=encoded_dataset, shuffle=True, collate_fn=collate_train, batch_size=batch_size)

  for label_type in label_types:
    
    optimizer = torch.optim.AdamW(model.classifiers[label_type].parameters(), lr=5e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*len(train_dataloader))
    for _ in NUM_EPOCHS:
      
      model.classifiers[label_type].train()
      for batch in train_dataloader:
        batch = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch.items()} # ensure every tensor is on the same device as the model
        optimizer.zero_grad() # zero out all the gradients each batch for better training
        
        labels = batch["labels_dict"][label_type]
        weights = weight_dict[label_type]
        input_mask = batch["input_mask_dict"][label_type]
        
        logits, loss = model(batch["input_ids"], batch["bbox"], batch["attention_mask"], input_mask, labels, batch["pixel_values"], weights, label_type)
        if loss is None:
          continue
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        del logits
        del loss
        torch.cuda.empty_cache()
    
      model.save(model_dir)
      model.classifiers[label_type].eval()
    

if __name__ == "__main":
  params = init_params()
  model_dir = params.model_dir
  datafile = params.datafile
  label_types = ast.literal_eval(params.label_types)
  batch_size = int(params.batch_size)
  
  model = MultimodalLMV3ForTokenClass(label_types, model_dir=model_dir, device=DEVICE)
  processor = model.processor
  model.to(DEVICE)

  while True:
    # read in data from parent program check if pipe has broken as well, quit if this is the case
    data_type = sys.stdin.readline().strip()
    if data_type == '':
      break
    data_file = sys.stdin.readline().strip()
    if data_file == '':
      break
    with open(data_file, 'r') as df:
      data = json.load(df)

    # depending on type of data given review the data or train the AI on it
    if data_type == "review":
      reviewed_data = review_data(data)
      with open(data_file) as df:
        json.dump(reviewed_data, df)
    elif data_type == "train":
      train_data(data)
    else:
      print(f"invalid data type: {data_type}\n supported data types: review, train", file=sys.stderr)
  
    # print something to let parent know done training/reviewing
    sys.stdout.write("done\n")
    sys.stdout.flush()

  # once parent closses the pipe, die
  quit()
    