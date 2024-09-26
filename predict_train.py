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
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# since only supporting BIO type models can just set these as constant, no need to infer
NULL_TAG = "O"
TAG_TYPES = ["O", "B-", "I-"]
ID2TYPE = {i:t for i, t in enumerate(TAG_TYPES)}
TYPES2ID = {t:i for i, t in enumerate(TAG_TYPES)}

# training hyperparameters
NULL_PERCENT = 4
NUM_EPOCHS = 5


def init_params():
  parser = argparse.ArgumentParser(prog="review_train.py", description="review and/or train a given model on given data")
  parser.add_argument("model_dir")
  parser.add_argument("label_types")
  parser.add_argument("batch_size")
  return parser.parse_args()


def reduce_bbox(bbox, width, height):
  # check if box is out of bounds at all ad adjust if necessary
  if bbox[0] > width:
    bbox[0] = width
  if bbox[1] > height:
    bbox[1] = height
  if bbox[2] > width:
    bbox[2] = width
  if bbox[3] > height:
    bbox[3] = height

  if bbox[0] < 0:
    bbox[0] = 0
  if bbox[1] < 0:
    bbox[1] = 0
  if bbox[2] < 0:
    bbox[2] = 0
  if bbox[3] < 0:
    bbox[3] = 0
  
  return bbox


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
  bboxs = [reduce_bbox(bbox, image.width, image.height) for bbox in example["bboxs"]]
  bboxs = [normalize_bbox(bbox, image.width, image.height) for bbox in bboxs]


  with torch.no_grad():
    encoded_inputs = processor(
      image, words, boxes=bboxs, padding="max_length",
      truncation=True, return_tensors="pt"
    ).to(DEVICE)
  
  encoded_inputs["word_ids"] = encoded_inputs.word_ids()
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
" PREDICT METHOD: ....
" ------------------------------------------------------------ """
def predict_data(data, batch_size, model, processor, label_types):
  dataset = Dataset.from_list(data)
  encoded_dataset = dataset.map(encode_review_dataset, fn_kwargs={"processor":processor}, batched=False)
  eval_dataloader = DataLoader(encoded_dataset, batch_size, shuffle=False, collate_fn=collate_review)
  model.to(DEVICE)
  
  for i, batch_collate in enumerate(eval_dataloader):
    batch_collate = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch_collate.items()} # ensure every tensor is on the same device as the model

    with torch.no_grad():
      for label_type in label_types:
        logits, loss = model(**batch_collate, label_type=label_type)
        if logits is None:
          continue
        predictions = logits.argmax(-1).squeeze()
        #print(predictions, file=sys.stderr)
        for j, prediction in enumerate(predictions):
          prediction = [p.item() for p in prediction]
          print(i*batch_size+j, file=sys.stderr)
          prediction = align_token_to_word(prediction, encoded_dataset["word_ids"][i*batch_size+j])
          true_prediction = [f"{ID2TYPE[p]}{label_type}" if ID2TYPE[p] != NULL_TAG else NULL_TAG for p in prediction]
          data[i*batch_size+j]["ner_tags"][label_type] = true_prediction
        del logits
        del predictions
    
    del batch_collate
    torch.cuda.empty_cache()
  
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


def encode_train_dataset(example, label_types=None, processor=None):
  image = Image.open(example["image_path"]).convert("RGB")
  words = example["words"]
  bboxs = [normalize_bbox(bbox, image.width, image.height) for bbox in example["bboxs"]]

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
  input_mask = [{label_type:0 for label_type in label_types} for example in training_data]
  null_cnt_dict = {label_type:0 for label_type in label_types} # make a null counter for each label type
  allowed_null_ex = {label_type:0 for label_type in label_types} # keep track of how many null examples are allowed for each label type

  # mask any examples that are not null
  for i, example in enumerate(training_data):
    
    ner_tags_dict = example["ner_tags"] # dict of all the kinds of tags
    for label_type in label_types:
      tags = ner_tags_dict[label_type]
      tags = [TYPES2ID[tag[:2]] for tag in tags] # convert tags to numbers

      if max(tags) != 0:
        null_cnt_dict[label_type] += 1
        if null_cnt_dict[label_type] >= NULL_PERCENT:
          allowed_null_ex[label_type] += 1
          null_cnt_dict[label_type] = 0
        input_mask[i][label_type] = 1

  # debugging print
  for label_type in label_types:
    total_masked = 0
    for mask_dict in input_mask:
      mask = mask_dict[label_type]
      total_masked += mask
    print(f"{label_type}: {total_masked}", file=sys.stderr)

  # mask a given percentage of the remaining null examples
  shuffled_data = [(i, example) for i, example in enumerate(training_data)]
  shuffled_data = random.sample(shuffled_data, len(training_data))
  for i, example in shuffled_data:
    ner_tags_dict = example["ner_tags"] # dict of all the kinds of tags
    for label_type in label_types:
      tags = ner_tags_dict[label_type]
      tags = [TYPES2ID[tag[:2]] for tag in tags] # convert tags to numbers

      if max(tags) == 0 and allowed_null_ex[label_type] > 0:
        allowed_null_ex[label_type] -= 1
        input_mask[i][label_type] = 1
  
  # debugging print
  for label_type in label_types:
    total_masked = 0
    for mask_dict in input_mask:
      mask = mask_dict[label_type]
      total_masked += mask
    print(f"{label_type}: {total_masked}", file=sys.stderr)

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
def train_data(data, batch_size, model, processor, model_dir, label_types):
  dataset = Dataset.from_list(data)
  input_mask = create_input_mask(data, label_types)
  dataset = dataset.add_column("input_mask", input_mask)
  weight_dict = generate_normalized_weights(dataset, label_types)
  print(weight_dict, file=sys.stderr)
  encoded_dataset = dataset.map(encode_train_dataset, fn_kwargs={"label_types":label_types, "processor":processor}, batched=False)

  #train_dataloader = DataLoader(dataset=encoded_dataset, shuffle=True, collate_fn=collate_train, batch_size=batch_size)

  for label_type in label_types:
    
    # filter the dataset so that only the masked examples for this type are present
    filtered_dataset = encoded_dataset.filter(lambda example: example["input_mask"][label_type] == 1)
    train_dataloader = DataLoader(dataset=filtered_dataset, shuffle=True, collate_fn=collate_train, batch_size=batch_size)

    print(f"---------------------{label_type}----------------------", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.classifiers[label_type].parameters(), lr=5e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=NUM_EPOCHS*len(train_dataloader)*0.1, num_training_steps=NUM_EPOCHS*len(train_dataloader))
    model.classifiers[label_type].train()
    model.classifiers[label_type].to(model.device)
    for i in range(NUM_EPOCHS):
      print(f"---------------------EPOCH-{i}------------------------", file=sys.stderr)
      
      for j, batch in enumerate(train_dataloader):
        print(f"==Batch={j}/{len(train_dataloader)}==", file=sys.stderr)
        optimizer.zero_grad() # zero out all the gradients each batch for better training
        batch = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch.items()} # ensure every tensor is on the same device as the model
        
        labels = batch["labels_dict"][label_type]
        weights = weight_dict[label_type]
        input_mask = batch["input_mask_dict"][label_type]
        if input_mask.max().item() == 0:
          optimizer.step()
          lr_scheduler.step()
          continue
        print(f"input mask: {input_mask}", file=sys.stderr)
        
        logits, loss = model(batch["input_ids"], batch["bbox"], batch["attention_mask"], input_mask, labels, batch["pixel_values"], weights, label_type)
        if loss is None:
          continue
        print(f"batch_loss: {loss.item()}", file=sys.stderr)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        del logits
        del loss
        torch.cuda.empty_cache()
    
      model.save(model_dir)
      torch.cuda.empty_cache()

    del filtered_dataset
    del train_dataloader
    model.classifiers[label_type].to('cpu')
    torch.cuda.empty_cache()



if __name__ == "__main__":
  sys.stderr.write("program launched:")
  params = init_params()
  model_dir = params.model_dir
  label_types = ast.literal_eval(params.label_types)
  batch_size = int(params.batch_size)
  print(model_dir, label_types, batch_size, file=sys.stderr)
  
  model = MultimodalLMV3ForTokenClass(label_types, model_dir=model_dir, device=DEVICE)
  processor = model.processor

  while True:
    # read in data from parent program check if pipe has broken as well, quit if this is the case
    data_type = sys.stdin.readline().strip()
    print(f"got data type {data_type}", file=sys.stderr)
    if data_type == '':
      break
    data_file = sys.stdin.readline().strip()
    print(f"got data file {data_file}", file=sys.stderr)
    if data_file == '':
      break
    with open(data_file, 'r') as df:
      data = json.load(df)
      print("loaded data in pred/train", file=sys.stderr)
      print(len(data), file=sys.stderr)
      df.close()

    # depending on type of data given review the data or train the AI on it
    if data_type == "predict":
      reviewed_data = predict_data(data, batch_size, model, processor, label_types)
      print('done reviewing', file=sys.stderr)
      with open(data_file, 'w') as df:
        json.dump(reviewed_data, df, indent=2)
        df.close()
    elif data_type == "train":
      train_data(data, batch_size, model, processor, model_dir, label_types)
    else:
      print(f"invalid data type: {data_type}\n supported data types: predict, train", file=sys.stderr)
  
    # print something to let parent know done training/reviewing
    sys.stdout.write("done\n")
    sys.stdout.flush()

  # once parent closses the pipe, die
  quit()
    