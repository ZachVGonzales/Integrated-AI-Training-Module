from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path


CHECKPOINT = "microsoft/layoutlmv3-base"
TAG_TYPES = ['O', 'B-', 'I']
TYPE2ID = {i:t for i, t in enumerate(TAG_TYPES)}



"""
Params:
 - label_types: user provided list of label types that the model shall make predictions on
 - base_checkpoint: the directory or huggingface repo where the base pretrained model / processor can be loaded from
 - model_dir: the directory where the exisiting classifiers can be loaded from 
"""
class MultimodalLMV3ForTokenClass(nn.Module):
  def __init__(self, label_types: list, base_checkpoint: str = "microsoft/layoutlmv3-base", model_dir: str | None = None, device=None) -> None:
    super(MultimodalLMV3ForTokenClass, self).__init__()
    self.classifiers = {}
    self.label_types = label_types
    self.num_labels = 3 # for BIO tagging this number is not variable so defined here
    self.processor = LayoutLMv3Processor.from_pretrained(base_checkpoint, apply_ocr=False)
    self.device = device
    
    if model_dir is not None:
      for label_type in label_types:
        classifier_path = f"{model_dir}/{label_type}"
        print(f"loading {classifier_path}", file=sys.stderr)
        if os.path.exists(classifier_path):
          self.classifiers[label_type] = LayoutLMv3ForTokenClassification.from_pretrained(classifier_path, local_files_only=True, num_labels=self.num_labels)
        else:
          self.classifiers[label_type] = LayoutLMv3ForTokenClassification.from_pretrained(base_checkpoint, num_labels=self.num_labels)
    else:
      self.classifiers = {label_type: LayoutLMv3ForTokenClassification.from_pretrained(base_checkpoint, num_labels=self.num_labels) for label_type in self.label_types}

  def forward(self, 
              input_ids: torch.Tensor | None = None, 
              bbox: torch.Tensor | None = None, 
              attention_mask: torch.Tensor | None = None, 
              labels: torch.Tensor | None = None,
              pixel_values: torch.Tensor | None = None,
              weights: torch.Tensor | None = None,
              label_type: str | None = None) -> tuple:

    # only want the logits from the model so we can weight loss function ourselves to account for imbalanced set
    classifier = self.classifiers[label_type]
    try:
      logits = classifier(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values).logits
    except Exception as e:
      print(f"exception {e}", file=sys.stderr)
      torch.set_printoptions(threshold=10000)
      print(input_ids.shape, input_ids.max(), input_ids.min(), input_ids, file=sys.stderr)
      print(bbox.shape, bbox, file=sys.stderr)
      print(attention_mask.shape, attention_mask, file=sys.stderr)
      print(pixel_values.shape, pixel_values, file=sys.stderr)
      return None, None

    if labels is not None and weights is not None:
      loss = self.compute_loss(logits, labels, weights)
      return logits, loss
  
    return logits, None

  
  def compute_loss(self, logits, labels, weights):
    aligned_labels = labels.view(-1) # shape (batch_size * seq_len)
    aligned_logits = logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_classes: 3)
    
    weighted_loss_fct = nn.CrossEntropyLoss(weight=weights)
    loss = weighted_loss_fct(aligned_logits, aligned_labels)
    return loss
  

  def to(self, device):
    for classifier in self.classifiers.values():
      classifier.to(device)


  def get_parameters(self):
    grouped_parameters = []
    for classifier in self.classifiers.values():
      grouped_parameters.extend(classifier.parameters())
    return grouped_parameters


  def save(self, model_dir: str) -> None:
    for label_type, classifier in self.classifiers.items():
      classifier_path = f"{model_dir}/{label_type}"
      if not os.path.exists(classifier_path):
        os.makedirs(classifier_path)
      classifier.save_pretrained(classifier_path)