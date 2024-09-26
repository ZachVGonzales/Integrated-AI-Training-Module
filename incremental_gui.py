import tkinter as tk
from tkinter import Tk
from annotation_app import AnnotationApp
import json
import sys
from PIL import Image, ImageTk
import argparse
import ast


class IncrementalGui(AnnotationApp):
  def __init__(self, root: Tk, page_lim: int = 10, label_types: list = ..., json_file: str = "annotations.json") -> None:
    super().__init__(root, page_lim, label_types)

    self.json_file = json_file
    self.export_ready = False

    # extra review specific UI buttons
    self.btn_del_all = tk.Button(self.bottom_frame, text="DEL ALL", command=self.delete_all)
    self.btn_del_all.pack(side=tk.LEFT)

    self.btn_export = tk.Button(self.bottom_frame, text="Export Data", command=self.export_dataset)
    self.btn_export.pack(side=tk.LEFT)

    # prediction / training switches
    self.prediction_var = tk.BooleanVar(value=False)
    self.prediction_button = tk.Button(self.bottom_frame, text="Generate Predictions", bg="red", command=self.toggle_prediction, width=8)
    self.prediction_button.pack(padx=5)

    self.train_var = tk.BooleanVar(value=False)
    self.train_button = tk.Button(self.bottom_frame, text="Train Model", bg="red", command=self.toggle_train, width=8)
    self.train_button.pack(padx=5)
  

  """
  translate a list of ner_tags into 2 parallel dictionary's of words/indexs for each label type
  """
  def convert_tags_to_annotations(self, ner_tags, words):
    annotation_words = {label:[] for label in self.label_types}
    annotation_idxs = {label:[] for label in self.label_types}
    
    for label_type in self.label_types:
      for idx, tag in enumerate(ner_tags[label_type]):
        if tag == self.ner_types[0]: # if the tag is null then skip
          continue

        ner_id = self.nertype2id[tag]
        
        if ner_id % 2 or not annotation_idxs[label_type]: # if it is a B- tag or the first I- tag (before a B- tag has appeared)
          annotation_words[label_type].append([])         # then start a new group for that label
          annotation_idxs[label_type].append([])

        annotation_words[label_type][-1].append(words[idx])
        annotation_idxs[label_type][-1].append(idx)
    
    return (annotation_words, annotation_idxs)

  """
  load images and image words from user provided pdf documents using pymupdf
  """
  def load_pages(self):
    # write the prediction setting to the parent
    if self.prediction_var.get():
      sys.stdout.write("True\n")
    else:
      sys.stdout.write("False\n")
    sys.stdout.flush()

    # wait for AI process to predict and give the input/output files
    self.json_file = sys.stdin.readline().strip()
    with open(self.json_file, 'r') as datafile:
      dataset = json.load(datafile)
      datafile.close()  
    
    for example in dataset:
      doc_name = example["doc_name"]
      page_num = example["doc_page"]
      image_path = example["image_path"]
      words = example["words"]
      bboxs = example["bboxs"]
      ner_tags = example["ner_tags"] # dict where key = label_type, value = ner_tags for given type

      self.page_images.append(image_path)
      self.page_ids.append((doc_name, int(page_num)))
      self.words.append(words)
      self.bboxs.append(bboxs)
      self.ner_tags.append(ner_tags)
      self.annotations.append(self.convert_tags_to_annotations(ner_tags, words))

    self.export_ready = True
    self.load_page() # load the first page to start editing

  """
  load the image at the current index into view for the user
  """
  def load_page(self):
    # if the image list is empty don't do anything
    if not self.page_images:
      return
    
    image_path = self.page_images[self.current_page_index]
    image = Image.open(image_path)
    
    # store the info for just the current page
    self.current_page_words = self.words[self.current_page_index]
    self.current_page_bboxs = self.bboxs[self.current_page_index]
    self.current_ner_tags = self.ner_tags[self.current_page_index]
    self.current_annotation_words = self.annotations[self.current_page_index][0]
    self.current_annotation_idxs = self.annotations[self.current_page_index][1]
    self.photo = ImageTk.PhotoImage(image) # save a reference to the photo object

    self.left_canvas.config(width=image.width, height=image.height) # change shape to match photo
    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo) # add image to canvas
    self.display_current_annotations()

  """
  export the current dataset now that it has been reviewed
  """
  def export_dataset(self):
    if not self.export_ready:
      return

    self.save_page()

    doc_names, doc_pages = zip(*self.page_ids)
    reviewed_dataset = []
    for doc_name, doc_page, words, bboxs, image_path, ner_tags, annotations in zip(doc_names, doc_pages, self.words, self.bboxs, self.page_images, self.ner_tags, self.annotations):
      reviewed_dataset.append({"doc_name": doc_name, "doc_page": doc_page, "words": words, "bboxs": bboxs, "image_path":image_path, "ner_tags": ner_tags})
    
    with open(self.json_file, 'w') as datafile:
      json.dump(reviewed_dataset, datafile)
      datafile.close()

    if self.train_var.get():
      sys.stdout.write("True")
    else:
      sys.stdout.write("False")
    sys.stdout.flush()

    self.reset_app()
    self.export_ready = False

  """
  delete all annotations on the given page
  """
  def delete_all(self):
    self.current_ner_tags = {label_type: [self.ner_types[0] for _ in self.current_page_words] for label_type in self.label_types} 
    self.current_annotation_words = {label: [] for label in self.label_types}
    self.current_annotation_idxs = {label: [] for label in self.label_types} 

    self.selected_words = []
    self.selected_idxs = [] 
    self.selected_annotation_idx = None 
    self.selected_rect_ids = [] 

    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_current_box = None

    self.display_current_annotations()

  """
  resets the app to it's original state (for when batch is exported)
  """
  def reset_app(self):
    self.page_images = [] 
    self.annotations = [] 
    self.page_ids = []
    self.words = [] 
    self.bboxs = [] 
    self.ner_tags = []

    self.current_page_index = 0 
    self.current_page_words = [] 
    self.current_page_bboxs = [] 
    self.current_ner_tags = []
    self.current_annotation_words = {} 
    self.current_annotation_idxs = {} 
    self.photo = None 

    self.selected_words = []
    self.selected_idxs = [] 
    self.selected_annotation_idx = None 
    self.selected_rect_ids = [] 

  """
  toggle the state of the prediction variable
  """
  def toggle_prediction(self):
    if self.prediction_var.get():
      self.prediction_var.set(False)
      self.prediction_button.config(bg="red")
    else:
      self.prediction_var.set(True)
      self.prediction_button.config(bg="green")

  """
  toggle the state of the train variable
  """
  def toggle_train(self):
    if self.train_var.get():
      self.train_var.set(False)
      self.train_button.config(bg="red")
    else:
      self.train_var.set(True)
      self.train_button.config(bg="green")


def init_params():
  parser = argparse.ArgumentParser(prog="review_train.py", description="review and/or train a given model on given data")
  parser.add_argument("label_types")
  return parser.parse_args()



if __name__ == "__main__":
  params = init_params()
  label_types = ast.literal_eval(params.label_types)
  root = tk.Tk()
  app = IncrementalGui(root, label_types=label_types)
  root.mainloop()