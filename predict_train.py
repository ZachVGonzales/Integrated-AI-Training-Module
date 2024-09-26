from custom_models import MultimodalLMV3ForTokenClass, VisionSequenceClassifiers
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
import nltk # for synonym replacement
from nltk.corpus import wordnet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# since only supporting BIO type models can just set these as constant, no need to infer
NULL_TAG = "O"
TAG_TYPES = ["O", "B-", "I-"]
ID2TYPE = {i:t for i, t in enumerate(TAG_TYPES)}
TYPES2ID = {t:i for i, t in enumerate(TAG_TYPES)}

# training hyperparameters
NULL_PERCENT = 4
NUM_EPOCHS = 5
DUP_LIMIT = 3 # the maximum number of times that norm examples can be duplicated


def init_params():
  parser = argparse.ArgumentParser(prog="review_train.py", description="review and/or train a given model on given data")
  parser.add_argument("model_dir")
  parser.add_argument("label_types")
  parser.add_argument("batch_size")
  parser.add_argument("-s", "--pskip", action="store_true", required=False)
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


def convert_tags_to_sequences(ner_tags, words):
  sequences = []
  idxs = []
    
  for idx, tag in enumerate(ner_tags):
    tag = tag[:2]
    if tag == TAG_TYPES[0]: # if the tag is null then skip
      continue

    if tag == TAG_TYPES[1] or not sequences: # if it is a B- tag or the first I- tag (before a B- tag has appeared)
      sequences.append([])                   # then start a new group for that label
      idxs.append([])
    sequences[-1].append(words[idx])
    idxs[-1].append(idx)

  return sequences, idxs


""" --------------------------------------------------------------
" PREDICT METHOD: ....
" ------------------------------------------------------------ """
def predict_data(data, batch_size, first_degree_model: MultimodalLMV3ForTokenClass, first_degree_processor, label_types):
  dataset = Dataset.from_list(data)
  encoded_dataset = dataset.map(encode_review_dataset, fn_kwargs={"processor":first_degree_processor}, batched=False)
  eval_dataloader = DataLoader(encoded_dataset, batch_size, shuffle=False, collate_fn=collate_review)
  

  for i, example in enumerate(data):
    example["annotation_words"] = {label_type:[] for label_type in label_types}
    example["annotation_idxs"] = {label_type:[] for label_type in label_types}
    data[i] = example

  for label_type in label_types:
    print(f"-------------------------{label_type}---------------------------", file=sys.stderr)
    first_degree_model.classifiers[label_type].to(first_degree_model.device)
    
    for i, batch_collate in enumerate(eval_dataloader):
      batch_collate = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch_collate.items()} # ensure every tensor is on the same device as the model
      with torch.no_grad():
        logits, _ = first_degree_model(**batch_collate, label_type=label_type)
        if logits is None:
          continue
        
        predictions = logits.argmax(-1).squeeze()
        for j, prediction in enumerate(predictions):
          prediction = [p.item() for p in prediction]
          prediction = align_token_to_word(prediction, encoded_dataset["word_ids"][i*batch_size+j])
          true_prediction = [f"{ID2TYPE[p]}{label_type}" if ID2TYPE[p] != NULL_TAG else NULL_TAG for p in prediction]
          data[i*batch_size+j]["ner_tags"][label_type] = true_prediction
          print(f"prediction: {i*batch_size+j}/{len(eval_dataloader)*batch_size}", file=sys.stderr)
          sequences, idxs = convert_tags_to_sequences(true_prediction, data[i*batch_size+j]["words"])
          data[i*batch_size+j]["annotation_words"][label_type] = sequences
          data[i*batch_size+j]["annotation_idxs"][label_type] = idxs

        del logits
        del predictions
        del batch_collate
        torch.cuda.empty_cache()
  
  return data



# alignes the list of word labels/predictions to there coresponding tokens
def align_word_to_token(text, boxes, word_labels, tokenizer):
  token_labels = []
  word_ids = tokenizer(text=text, boxes=boxes, padding='max_length', truncation=True).word_ids()
  current_word_idx = None

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


def count_null_examples(dataset, label_type):
  num_null_ex = 0
  for example in dataset:
    mask = example["input_mask"][label_type]
    if mask == 0:
      num_null_ex += 1
  return num_null_ex


def encode_train_dataset(example, processor=None):
  image = Image.open(example["image_path"]).convert("RGB")
  words = example["words"]
  bboxs = [normalize_bbox(bbox, image.width, image.height) for bbox in example["bboxs"]]

  with torch.no_grad():
    encoded_inputs = processor(
      image, words, boxes=bboxs, padding="max_length",
      truncation=True, return_tensors="pt"
    ).to(DEVICE)

  labels = example["ner_tags"]

  labels = align_word_to_token(words, bboxs, labels, processor.tokenizer)
  labels = convert_token_tags_to_ids(labels)
  labels = torch.tensor(labels, device=DEVICE, dtype=torch.long)
  labels = labels.to(DEVICE)

  encoded_inputs["input_mask"] = example["input_mask"]
  encoded_inputs["labels"] = labels
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

  return input_mask


def generate_normalized_weights(dataset: list):
  weights = [0 for _ in TAG_TYPES]

  # find the frequencies of every tag in the dataset
  for example in dataset:
    ner_tags = example["ner_tags"]
    for tag in ner_tags:
      weights[TYPES2ID[tag[:2]]] += 1
  
  # compute weights for loss as inverse of frequencies
  total_tag_count = sum(weights)
  weights = [(total_tag_count / count) for count in weights]

  # Normalize the weights and convert to tensor format
  min_weight = min(weights)
  normalized_weights = [(weight / min_weight) for weight in weights]
  weight_tensor = torch.tensor(normalized_weights, dtype=torch.float)
  weight_tensor = weight_tensor.to(DEVICE)
  
  return weight_tensor


def collate_train(batch):
  input_ids = [torch.tensor(example["input_ids"]) for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
  bbox = [torch.tensor(example["bbox"]) for example in batch]
  pixel_values = [torch.tensor(example["pixel_values"]) for example in batch]
  labels = [torch.tensor(example["labels"]) for example in batch]

  # pad
  labels = pad_sequence(labels, batch_first=True, padding_value=-100)
  if len(labels.shape) > 2:
    labels = labels.squeeze()
  if len(labels.shape) == 1:
    labels = labels.unsqueeze(-1)
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

  return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "labels": labels, "bbox":bbox_padded, "pixel_values":pixel_values_padded}


def type_dataset(example, label_type):
  example["ner_tags"] = example["ner_tags"][label_type]
  example["input_mask"] = example["input_mask"][label_type]
  return example


def introduce_typo(words: list[str], p_typo=0.15):
  new_words = []
  for word in words:
    chars = list(word)
    if len(chars) > 1 and random.random() < p_typo:
      swap_pos = random.randint(0, len(chars)-2)
      chars[swap_pos], chars[swap_pos+1] = chars[swap_pos+1], chars[swap_pos]
      typo_word = ''.join(chars)
      new_words.append(typo_word)
    else:
      new_words.append(word)
  return new_words


# need nltk to use this
def rand_synonym_replacement(words, p_synonym=0.15):
  try: # check if wordnet is available, if not then do not attempt replacement
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
  except:
    return words

  new_words = []
  for word in words:
    synonyms = set()
    for syn in wordnet.synsets(word, lang='eng'):
      for lemma in syn.lemmas():
        synonyms.add(lemma.name().replace('_', ' '))
    if synonyms and random.random() < p_synonym:
      new_words.append(random.choice(list(synonyms)))
    else:
      new_words.append(word)
  return new_words


def augment_dataset(example):
  words = example["words"]
  if random.random() < 0.5:
    words = introduce_typo(words)
  else:
    words = rand_synonym_replacement(words)
  example["words"] = words
  return example


def reduce_dataset(dataset, size):
  if len(dataset) < size:
    return dataset
  shuffled_dataset = dataset.shuffle()
  reduced_dataset = shuffled_dataset.select(range(size))
  del dataset
  del shuffled_dataset
  return reduced_dataset


def generate_balanced_dataset(dataset, label_type, ratio):
  typed_dataset = dataset.map(type_dataset, fn_kwargs={"label_type":label_type}, batched=False)
  null_dataset = typed_dataset.filter(lambda example: example["input_mask"] == 0)
  norm_dataset = typed_dataset.filter(lambda example: example["input_mask"] == 1)
  del typed_dataset

  if ratio > DUP_LIMIT:
    ratio = DUP_LIMIT
    null_dataset = reduce_dataset(null_dataset, ratio*len(norm_dataset))
  
  balanced_dataset = Dataset.from_dict({col: null_dataset[col] + norm_dataset[col] for col in null_dataset.column_names})
  for _ in range(ratio):
    augmented_normal_examples = norm_dataset.map(augment_dataset, batched=False)
    balanced_dataset = Dataset.from_dict({col: balanced_dataset[col] + augmented_normal_examples[col] for col in balanced_dataset.column_names})
  balanced_dataset.remove_columns("input_mask") # don't need input mask anymore since dataset is balanced
  del norm_dataset
  del null_dataset
  del augmented_normal_examples
  return balanced_dataset


def encode_strain_dataset(example, tokenizer, stypes2id):
  encoded_inputs = tokenizer(example["sequence"], return_tensors="pt", padding="max_length", truncation=True, is_split_into_words=True)
  encoded_inputs["label"] = torch.tensor(stypes2id[example["label"]], dtype=torch.long)
  return encoded_inputs


def collate_strain(batch):
  input_ids = [torch.tensor(example["input_ids"]).squeeze() if type(example["input_ids"]) != torch.Tensor else example["input_ids"] for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]).squeeze() if type(example["attention_mask"]) != torch.Tensor else example["attention_mask"] for example in batch]
  labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)

  input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
  attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

  return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


"""
NOTE: this must be changed if the model I end up selecting changes
"""
def train_data(data, batch_size, first_degree_model, first_degree_processor, model_dir, label_types, skip_primary=False):
  dataset = Dataset.from_list(data)
  input_mask = create_input_mask(data, label_types)
  dataset = dataset.add_column("input_mask", input_mask)

  # train a classifier for each data type
  for label_type in label_types:
    print(f"---------------------{label_type}----------------------", file=sys.stderr)

    # create a sub dataset for this label type and balance the dataset by expanding and augmenting
    num_null_ex = count_null_examples(dataset, label_type)
    num_norm_ex = len(dataset) - num_null_ex
    ratio = num_null_ex / num_norm_ex
    ratio = ratio.__ceil__()
    balanced_dataset = generate_balanced_dataset(dataset, label_type, ratio)
    weights = generate_normalized_weights(balanced_dataset)
    balanced_dataset = balanced_dataset.map(encode_train_dataset, fn_kwargs={"processor":first_degree_processor}, batched=False)
    print(f"balanced dataset: {balanced_dataset}, {len(balanced_dataset)}", file=sys.stderr)
    
    # initialize the dataloader, optimizer, and learning rate scheduler
    train_dataloader = DataLoader(dataset=balanced_dataset, shuffle=True, collate_fn=collate_train, batch_size=batch_size)
    optimizer = torch.optim.AdamW(first_degree_model.classifiers[label_type].parameters(), lr=5e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=NUM_EPOCHS*len(train_dataloader)*0.1, num_training_steps=NUM_EPOCHS*len(train_dataloader))
    
    # move the correct model to the device currently using and enter train mode
    first_degree_model.classifiers[label_type].train()
    first_degree_model.classifiers[label_type].to(first_degree_model.device)
    
    for i in range(NUM_EPOCHS):
      if skip_primary:
        continue
      print(f"---------------------EPOCH-{i}------------------------", file=sys.stderr)
      
      for j, batch in enumerate(train_dataloader):
        optimizer.zero_grad() # zero out all the gradients each batch for better training
        batch = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch.items()} # ensure every tensor is on the same device as the model
        torch.set_printoptions(threshold=10000)
        print(f"batch_labels: {batch['labels']}", file=sys.stderr)
        torch.set_printoptions(threshold=100)

        logits, loss = first_degree_model(batch["input_ids"], batch["bbox"], batch["attention_mask"], batch["labels"], batch["pixel_values"], weights, label_type)
        
        if loss is None:
          optimizer.step()
          lr_scheduler.step()
          continue
        print(f"batch_loss ({j}/{len(train_dataloader)}): {loss.item()}\n\n", file=sys.stderr)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        del logits
        del loss
        torch.cuda.empty_cache()

      first_degree_model.save(model_dir)
      torch.cuda.empty_cache()

    first_degree_model.classifiers[label_type].eval()
    first_degree_model.classifiers[label_type].to('cpu')
    del balanced_dataset
    del train_dataloader
    torch.cuda.empty_cache()


if __name__ == "__main__":
  sys.stderr.write("program launched:")
  params = init_params()
  model_dir = params.model_dir
  label_types = ast.literal_eval(params.label_types)
  batch_size = int(params.batch_size)
  skip_primary = params.pskip

  first_degree_classifiers = MultimodalLMV3ForTokenClass(label_types, model_dir=model_dir, device=DEVICE)
  first_degree_processor = first_degree_classifiers.processor

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
      predicted_data = predict_data(data, batch_size, first_degree_classifiers, first_degree_processor, label_types)
      print('done predicting', file=sys.stderr)
      with open(data_file, 'w') as df:
        json.dump(predicted_data, df, indent=2)
        df.close()
    elif data_type == "train":
      nltk.download('wordnet') # need to make sure wordnet is downloaded when training so can use synonym replacement
      nltk.download('omw-1.4') # added multilingual support
      train_data(data, batch_size, first_degree_classifiers, first_degree_processor, model_dir, label_types, skip_primary)
    else:
      print(f"invalid data type: {data_type}\n supported data types: predict, train", file=sys.stderr)
  
    # print something to let parent know done training/reviewing
    sys.stdout.write("done\n")
    sys.stdout.flush()

  # once parent closses the pipe, die
  quit()