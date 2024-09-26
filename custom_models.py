from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Model, LayoutLMv3Processor
import torch
import torch.nn as nn
import os
import sys


CHECKPOINT = "microsoft/layoutlmv3-base"
TAG_TYPES = ['O', 'B-', 'I']
TYPE2ID = {i:t for i, t in enumerate(TAG_TYPES)}


# used to aggregate all the hidden states of query tokens for a given query word
class AttentionAggregation(nn.Module):
  def __init__(self, hidden_size) -> None:
    super(AttentionAggregation, self).__init__()
    self.attention = nn.Linear(hidden_size, 1, bias=False)

  # note for shapes of inputs:
  # hidden_states = shape (batch_size, seq_length, hidden_size)
  # query_indices = shape (batch_size, 2) where 2 is for start and end of each query
  # attention mask = shape (batch_size), each element is a 1 (if query is valid) or 0 (if query is invalid)
  def forward(self, hidden_states: torch.Tensor, query_indices: torch.Tensor, attention_mask: torch.Tensor):
    batch_size, seq_length, hidden_size = hidden_states.shape

    # expand attention mask to fit shape of hidden states (this will nullify any invaild queries)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_length, hidden_size) 
    hidden_states = hidden_states * attention_mask # masks any hidden states that should not be used
    
    # since queries are not all same length need to expand, pad and mask along query range dimension
    max_query_len = max(query_indices[i, :][1].item() + 1 - query_indices[i, :][0].item() for i in query_indices.size(0)) # returns an int that is the length of longest range (inclusive)
    query_indices_expanded = torch.zeros(batch_size, max_query_len)
    for i in range(query_indices.size(0)): # go through each batch
      query_range = query_indices[i, :] # this is tensor of shape: (2)
      query_len = query_range[1].item() - query_range[0].item() + 1 # get the current queries length
      padding = max_query_len - query_len # the padding needed to add is 
      expanded_range = torch.arange(query_range[0].item(), query_range[1].item()) # shape: (query_len)
      expanded_range = nn.functional.pad(expanded_range, (0, padding), "constant", -1) # pad with zeros to shape (query_max_len)
      query_indices_expanded[i, :] = expanded_range # replace row with expanded range

    mask = (query_indices_expanded == -1) # create a mask for the padded values
    query_indices_expanded[mask] = 0 # replace -1's with 0s so can use in gather without error
    query_indices_expanded = query_indices_expanded.unsqueeze(-1).expand(-1, -1, hidden_size) # now of shape (batch_size, max_query_len)
    query_hidden_states = torch.gather(hidden_states, dim=1, index=query_indices_expanded) # gather indices of sequence given in query indices
    query_hidden_states[mask.unsqueeze(-1).expand(-1, -1, hidden_size)] = 0 # mask what were originally pad values so dont get included in aggrigation, shape (batch_size, max_querry_len, hidden_size)

    attention_weights = torch.softmax(self.attention(query_hidden_states), dim=1) # feed all query indices through layer, should weight these appropriatly, apply softmax so wieghts sum to 1
    aggregated_hidden = torch.sum(attention_weights * query_hidden_states, dim=1) # apply the weights to each of the hidden states and sum the states together to get one aggrigated state
    return aggregated_hidden # shape: (batch_size, hidden_size)


# used for predicting the group of words belonging to a given query word
class GroupPredictionHead(nn.Module):
  def __init__(self, hidden_size) -> None:
    super(GroupPredictionHead, self).__init__()
    self.hidden_size = hidden_size
    
    # takes in a the vectors of the hidden state and query hidden state outputs class prediction
    self.binary_classifier = nn.Linear(hidden_size*2, 1) 
    self.sigmoid = nn.Sigmoid() # logit -> classificaiton predication
  
  # note for shape of inputs:
  # aggregated_query_hidden_state = shape (batch_size, hidden_size)
  # token_hidden_states = shape(batch_size, seq_length, hidden_size)
  def forward(self, aggregated_query_hidden_state, token_hidden_states):
    batch_size, seq_len, hidden_size = token_hidden_states.size() # get the dimensions of the hidden states
    query_hidden_expanded = aggregated_query_hidden_state.unsqueeze().expand(-1, seq_len, -1) # repeat qhs for as many tokens exist
    
    # concatinate query hidden state with every token hidden state
    classifier_inputs = torch.cat((query_hidden_expanded, token_hidden_states), dim=-1) 

    # pass the inputs through the classification layer and make predictions
    logits = self.binary_classifier(classifier_inputs)
    logits = logits.squeeze(-1)
    probabilities = self.sigmoid(logits)
    return probabilities


class LayoutLMv3ForMultiSpanClassification(nn.Module):
  def __init__(self, num_labels) -> None:
    super(LayoutLMv3ForMultiSpanClassification, self).__init__()
    self.LayoutLMv3 = LayoutLMv3ForTokenClassification.from_pretrained(CHECKPOINT, num_labels=num_labels)  # the base transformer model (outputs hidden states)
    self.hidden_size = self.LayoutLMv3.config.hidden_size
    self.aggregation_layer = AttentionAggregation(self.hidden_size) # agregates the query word tokens together into a single tensor
    self.group_prediction_head = GroupPredictionHead(self.hidden_size) # uses query tensor with all context word tensors to make predictions for each

  # note for shapes of inputs:
  # query_indices = shape (batch_size, max_query_num, 2) where 2 is for start and end of each query
  # query_attention_mask = shape (batch_size, max_query_num)
  # label_mask = shape (batch_size, seq_length, hidden_size)
  # word_ids (batch_size, seq_length)
  # other notes:
  # max_query_len is the maximum number of queries in one example in the batch
  def forward(self, input_ids, bbox, pixel_values, attention_mask, token_type_ids, word_ids, query_indices=None, query_attention_mask=None, label_mask=None, task=None, batch_size=4, seq_length=512, max_query_num=None):
    outputs = self.LayoutLMv3(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
    
    if task == "Classify": # in training classification and grouping are trained seperately so need different tasks for each
      return outputs.logits # tensor of shape (batch_size, seq_length, num_labels)
  
    elif task == "Group": # want to output tensor of shape (batch_size, max_query_num, seq_length, 2) where 2 is the number of labels (part of group, not part of group)
      probabilities = torch.zeros(batch_size, max_query_num, seq_length)
      all_hidden_states = outputs.hidden_states # this is a tuple of length config.hidden_layers whose contents are tensors of shape [batch_size, sequence_len, hidden_size]
      last_hidden_states = all_hidden_states[-1] # shape: (batch_size, seq_length, hidden_size)

      for i in range(query_indices.size(1)): # traverse queries one at a time 
        query_ind = query_indices[:, i, :] # query_ind now has shape: (batch_size, 2)
        mask = query_attention_mask[:, i] # mask now has shape: (batch_size)
        query_aggregate_state = self.aggregation_layer(last_hidden_states, query_ind, mask) # agrigate the query into one tensor representing the whole word
        group_prediction = self.group_prediction_head(query_aggregate_state, last_hidden_states * label_mask) # use query and only words in the label type
        probabilities[:, i, :] = group_prediction
      
      group_prediction = (group_prediction > 0.5).int() # turns the predictions into 0 or 1
      return group_prediction # shape (batch_size, max_num_queries, seq_length)
    
    elif task == "ClassifyAndGroup":
      probabilities = torch.zeros(batch_size, max_query_num, seq_length) # this will be the output tensor for group probibilities, shape: (batch_size, max_query_num, seq_length)
      all_hidden_states = outputs.hidden_states # this is a tuple of length config.hidden_layers whose contents are tensors of shape [batch_size, sequence_len, hidden_size]
      last_hidden_states = all_hidden_states[-1] # shape: (batch_size, seq_length, hidden_size)
      batch_size, seq_length, hidden_size = last_hidden_states.shape

      BIO_predictions = outputs.logits # tensor of shape (batch_size, seq_length, num_labels)
      label_mask = BIO_predictions.argmax(-1) # tensor of shape (batch_size, seq_length) where since the null label should always be at the first index of any label array the 0's should be ignored
      label_mask = (label_mask != 0).int() # change any label other than nulls to 1 so when tensor is crossed with this mask wont get any funny results
      label_mask = label_mask.unsqueeze(-1).expand(-1, -1, hidden_size) # now the mask matches shape (batch_size, seq_length, hidden_size)
      query_indices = BIO_predictions.argmax(-1)
      query_indices = (query_indices == 1).int() # just want the B-tag tokens, shape: (batch_size, seq_length)

      current_word_id = None
      query_lists = [[] for _ in range(batch_size)]
      for i in range(batch_size): # go through all batches
        for j in range(seq_length): # go through every token in each batch
          if query_indices[i][j] and current_word_id != word_ids[i][j]: # if the token is a valid query token and it is part of a new word token is start of new query
            query_lists[i].append([j, j])
          elif query_indices[i][j] and current_word_id == word_ids[i][j]: # if token is valid query token and part of same word then this is the end
            query_lists[i][-1][1] = j
      max_query_num = max(len(query_list) for query_list in query_lists) # find the longest list of queries
      query_indices = [query_list + [[-1] * len(query_list[0]) * (max_query_num-len(query_list))] for query_list in query_lists] # pad the lists so they are all the same length
      query_indices = torch.tensor(query_indices) # tensor now of shape: (batch_size, max_query_num, 2)
      query_attention_mask = query_indices[:, :, 0] # just need first index of query to check if query is null, shape: (batch_size, max_query_num)
      query_attention_mask = (query_attention_mask != -1).int() # 1 if not null, 0 if null

      for i in range(query_indices.size(1)): # traverse queries one at a time
        query_ind = query_indices[:, i, :] # query_ind shape: (batch_size, 2)
        mask = query_attention_mask[:, i] # mask shape: (batch_size)
        query_aggregate_state = self.aggregation_layer(last_hidden_states, query_ind, mask)
        group_prediction = self.group_prediction_head(query_aggregate_state, last_hidden_states * label_mask) # use query and only words in the label type
        probabilities[:, i, :] = group_prediction
      
      group_prediction = (group_prediction > 0.5).int() # turns the predictions into 0 or 1
      return group_prediction # shape (batch_size, max_num_queries, seq_length)
    
    else:
      return None # invalid task was assigned



class MultiLayeredHead(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MultiLayeredHead, self).__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    self.layer3 = nn.Linear(hidden_dim, output_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.relu(self.layer2(x))
    x = self.layer3(x)
    return x



class MultiHeadTokenClassification(nn.Module): 
  
  """
  Parameters:
  - label_types = list of strings representing label types
  - base_path = the path to the trained base of the model
  - trained_head_paths = dictionary of label_types and a trained head if it exists
  """
  def __init__(self, label_types, device=None, base_path="microsoft/layoutlmv3-base", trained_head_paths=None) -> None:
    super(MultiHeadTokenClassification, self).__init__()
    self.layoutlmv3_base = LayoutLMv3Model.from_pretrained(base_path) # load the base layout model
    self.config = self.layoutlmv3_base.config
    self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob) # set a dropout layer from configuration to prevent overfitting
    self.device = device

    # this is a dictionary of models, one for each type of label, accessed by the string for that label type
    self.classifier_heads = nn.ModuleDict({label_type: MultiLayeredHead(input_dim=int(self.config.hidden_size), hidden_dim=int(self.config.hidden_size / 2), output_dim=3) for label_type in label_types})

    if trained_head_paths is not None:
      for label_type, path in trained_head_paths.items():
        if label_type in self.classifier_heads: # should always be true unless caller is being dumb
          self.classifier_heads[label_type].load_state_dict(torch.load(path)) # load the trained head in place of the new one
        else:
          raise ValueError(f"No classification head found for label type: {label_type}")
  
  """
  defines a forward pass for the model, note only logits are returned and thus loss must be
  calculated in the training loop manually 
  """
  def forward(self, 
              input_ids=None, 
              bbox=None, 
              attention_mask=None, 
              input_mask=None,
              inputs_embeds=None,
              labels=None,
              return_dict=None,
              pixel_values=None,
              weights=None,
              words=None,        # TODO: remove "
              word_labels=None): # TODO: remove this param when done dubugging
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    #print(input_ids.shape, bbox.shape, attention_mask.shape, pixel_values.shape)

    outputs = self.layoutlmv3_base(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        return_dict=return_dict,
        pixel_values=pixel_values,
      )
    #print(f"outputs: {outputs[0].requires_grad}", file=sys.stderr)
    
    # we only want to grab the contextualized token embeddings not patch embeddings
    if input_ids is not None:
      input_shape = input_ids.size()
    else:
      input_shape = inputs_embeds.size()[:-1]
    
    seq_length = input_shape[1]
    # only take the text part of the output representations
    sequence_output = outputs[0][:, :seq_length] # now shape (batch_size, token_seq_len, hidden_size)
    sequence_output = self.dropout_layer(sequence_output)


    # now need to get logits from all classifier heads
    logits_dict = {}

    # go through each classifier and make prediction
    for label_type, classifier in self.classifier_heads.items():
      logits = classifier(sequence_output) # return shape: (batch_size, token_seq_len, 3)
      logits_dict[label_type] = logits * attention_mask.unsqueeze(-1) # mask the logits that were padded
    
    loss = None
    if labels is not None:
      words_str = [f"{i}. , {word}" for i, word in enumerate(words[0])]
      #print("words for ex 0 in batch:", words_str)
      #for label_type in labels.keys():
      #  print(f"word labels for {label_type}:", [f"{i}. , {label}" for i, label in enumerate(word_labels[0][label_type])])
      loss = self.compute_loss(logits_dict, labels, weights, input_mask)

    if loss is not None:
      return logits_dict, loss
    # output is a dict of a all label types logits, each value is (batch_size, token_seq_len, 3)
    return logits_dict

  """
  defines a method of saving the model that directly coresponds with how the model is loaded when 
  initialized.
  """
  def save(self, directory, processor = None):
    self.layoutlmv3_base.save_pretrained(directory) # save the base of the model
    
    for label_type, head in self.classifier_heads.items(): # save the head states individually
      head_path = f"{directory}{label_type}_head.pth"
      print(f"head saved at {head_path}")
      torch.save(head.state_dict(), head_path)
    
    if processor is not None:
      processor.save_pretrained(directory)
  
  """
  defines a method of loading the model that does not require a dictionary as input
  (unlike the init method)
  """
  @classmethod
  def load(cls, label_types, load_directory, device=None, load_processor = False):
    model = cls(label_types, base_path=load_directory, device=device)

    for label_type in label_types:
      head_path = f"{load_directory}{label_type}_head.pth"
      print(f"head_path: {head_path}")
      if os.path.exists(head_path):
        print(f"head_loaded for {label_type}")
        model.classifier_heads[label_type].load_state_dict(torch.load(head_path))

    if load_processor:
      processor = LayoutLMv3Processor.from_pretrained(load_directory, apply_ocr=False)
      return model, processor

    return model
  
  """
  computes the weighted loss for the model on a batch or just one example (should work for either)
  params: logits_dict - dictionary, key = label_type, value = tensor (batch_size, seq_length, 3)
          labels_dict - dictionary, key = label_type, value = tensor (batch_size, seq_length)
          weights - dictionary, key = label_type, value = tensor (3)
  """
  def compute_loss(self, logits_dict, labels_dict, weights_dict=None, input_mask_dict=None):
    loss = {}
    torch.set_printoptions(threshold=float(1024))
    for label_type, logits in logits_dict.items():
      labels = labels_dict[label_type]
      label_mask = input_mask_dict[label_type] # mask = shape (batch_size, 1)
      logit_mask = label_mask.unsqueeze(-1) # mask = shape (batch_size, 1 , 1)
      masked_labels = torch.where(label_mask == 1, labels, -100) # mask the labels applying -100 as the masked value
      masked_logits = logits * logit_mask # mask the logits

      labels = masked_labels.view(-1) # shape (batch_size * seq_len)
      logits = masked_logits.view(-1, 3) # shape (batch_size * seq_len, 3)

      weights = weights_dict[label_type]
      weighted_loss_fct = nn.CrossEntropyLoss(weight=weights)
      batch_loss = weighted_loss_fct(logits, labels)
      
      print(label_type)
      print(label_mask)
      #labels_str = [f"{num}, {TYPE2ID[tag.item()]}" if tag.item() != -100 else 'O' for num, tag in enumerate(masked_labels[0])]
      #print(f"{label_type} labels: {labels_str}")
      #print(masked_labels)
      #print(masked_logits)
      #print("batch loss:", batch_loss)
      #for batch_num in range(masked_logits.size(0)):
      #  logit = masked_logits[batch_num]
      #  label = masked_labels[batch_num]
      #  example_loss = weighted_loss_fct(logit, label)
      #  print("example loss", example_loss)
      #  print(logit)
      #  print(label)
      print(f"batch_loss for label type {label_type}: {batch_loss}")
      loss[label_type] = batch_loss
    print(f"loss: {loss}")
    return loss
        #if max(labels) == 0 and null_incl[label_type] != 0: # just skip calculating loss on examples that do not have any tags
        #  if null_incl[label_type] == 5:
        #    null_incl[label_type] = 0
        #  continue
        #null_incl[label_type] += 1
        

        #print(f"labels: {labels.shape} {labels}", file=sys.stderr)
        #print(f"logits: {logits.shape} {logits}", file=sys.stderr)
        #labels = labels.view(-1)
    
        #print(f"label_type: {label_type}", file=sys.stderr)
        #print(f"labels: {labels.shape} {labels}", file=sys.stderr)
        #print(f"logits: {logits.shape} {logits[:, 0]}", file=sys.stderr)

    if weights is not None:
      weighted_loss_fct = nn.CrossEntropyLoss(weight=weights)
    else:
      weighted_loss_fct = nn.CrossEntropyLoss()

    loss = weighted_loss_fct(logits, labels)
    
    #print(f"loss on example = {total_loss}", file=sys.stderr)
    return loss
  

#if __name__ == "__main__":
#  model = MultiHeadTokenClassification.load(["OBJ", "TR", "OE", "EM", "REV"], "../models/BIO-C")
#  processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
#  model.save("../models/BIO-C", processor)


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
              input_mask: torch.Tensor | None = None,
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

    if labels is not None and weights is not None and input_mask is not None:
      loss = self.compute_loss(logits, labels, weights, input_mask)
      return logits, loss
  
    return logits, None

  
  def compute_loss(self, logits, labels, weights, input_mask):
    masked_labels = torch.where(input_mask == 1, labels, -100) # set any masked inputs labels to the ignore index so loss is not calculated
    aligned_labels = masked_labels.view(-1) # shape (batch_size * seq_len)
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