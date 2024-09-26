from annotation_app import AnnotationApp
import tkinter as tk
import sys
import json
from PIL import Image, ImageTk
import argparse
import ast


class ReviewApp(AnnotationApp):
  def __init__(self, root: tk.Tk, page_lim: int = 10, label_types: list = ..., json_out: str = "annotations.json", classifications: dict = {"T_OBJ": ["performance", "cognative"], "E_OBJ": ["performance", "cognative"], "STEM": ["MC", "MA", "TF", "SA", "ES"]}) -> None:
    super().__init__(root, page_lim, label_types, json_out, classifications)

    # extra review specific UI buttons
    self.btn_del_all = tk.Button(self.bottom_frame, text="DEL ALL", command=self.delete_all)
    self.btn_del_all.pack(side=tk.LEFT)

    self.btn_export = tk.Button(self.bottom_frame, text="Export Data", command=self.export_dataset)
    self.btn_export.pack(side=tk.LEFT)

    self.json_file_in = json_out
    self.json_file_out = json_out
    self.export_ready = False
  

  """
  translate a list of ner_tags into 2 parallel dictionary's of words/indexs for each label type
  """
  def convert_tags_to_annotations(self, ner_tags, words, sequences=None, idxs=None, seq_classifications=None):
    annotation_words = {label:[] for label in self.label_types}
    annotation_idxs = {label:[] for label in self.label_types}
    annotation_classifications = {label:[] for label in self.classifications.keys()}
    
    for label_type in self.label_types:
      if sequences and idxs and (label_type in sequences.keys()): # instead of generating use what has already been generated
        annotation_words[label_type] = sequences[label_type]
        annotation_idxs[label_type] = idxs[label_type]
        if label_type in seq_classifications.keys():
          annotation_classifications[label_type] = seq_classifications[label_type]
        elif label_type in self.classifications.keys():
          annotation_classifications[label_type] = [self.classifications[label_type][0]*len(sequences[label_type])]
        continue

      for idx, tag in enumerate(ner_tags[label_type]):
        if tag == self.ner_types[0]: # if the tag is null then skip
          continue

        ner_id = self.nertype2id[tag]
        
        if ner_id % 2 or not annotation_idxs[label_type]: # if it is a B- tag or the first I- tag (before a B- tag has appeared)
          annotation_words[label_type].append([])         # then start a new group for that label
          annotation_idxs[label_type].append([])
          if label_type in self.classifications.keys():
            annotation_classifications[label_type].append(self.classifications[label_type][0])

        annotation_words[label_type][-1].append(words[idx])
        annotation_idxs[label_type][-1].append(idx)

    return (annotation_words, annotation_idxs, annotation_classifications)

  """
  load images and image words from user provided pdf documents using pymupdf
  """
  def load_pages(self):
    # wait for parent process to send the file for which to review 
    self.json_file_in = sys.stdin.readline().strip()
    self.json_file_out = sys.stdin.readline().strip()
    with open(self.json_file_in, 'r') as datafile:
      dataset = json.load(datafile)
      datafile.close()
    
    print(len(dataset), file=sys.stderr)

    for example in dataset:
      doc_name = example["doc_name"]
      abs_path = example["abs_path"]
      page_num = example["doc_page"]
      image_path = example["image_path"]
      words = example["words"]
      bboxs = example["bboxs"]
      images = example["context_images"]
      image_bboxs = example["image_bboxs"]
      ner_tags = example["ner_tags"] # dict where key = label_type, value = ner_tags for given type
      for label_type in self.label_types:
        if label_type not in ner_tags.keys():
          ner_tags[label_type] = [self.ner_types[0] for _ in words]
      
      # in the case that there already exist annotations use those but if not just ignore error and move on
      try:
        sequences = example["annotation_words"]
        idxs = example["annotation_idxs"]
        seq_labels = example["annotation_classifications"]
      except:
        sequences = None
        idxs = None
        seq_labels = None

      self.page_images.append(image_path)
      self.page_ids.append((doc_name, abs_path, int(page_num)))
      self.words.append(words)
      self.bboxs.append(bboxs)
      self.context_images.append(images)
      self.image_bboxs.append(image_bboxs)
      self.ner_tags.append(ner_tags)
      self.annotations.append(self.convert_tags_to_annotations(ner_tags, words, sequences, idxs, seq_labels))

    print(len(self.page_images), file=sys.stderr)

    self.export_ready = True
    self.load_page() # load the first page to start editing

  """
  load the image at the current index into view for the user
  """
  def load_page(self):
    # if the image list is empty don't do anything
    if not self.page_images:
      return
    
    print(f"page {self.current_page_index} / {len(self.page_images)}", file=sys.stderr)
    image_path = self.page_images[self.current_page_index]
    image = Image.open(image_path)
    
    # store the info for just the current page
    self.current_page_words = self.words[self.current_page_index]
    self.current_page_bboxs = self.bboxs[self.current_page_index]
    self.current_ner_tags = self.ner_tags[self.current_page_index]
    self.current_annotation_words = self.annotations[self.current_page_index][0]
    self.current_annotation_idxs = self.annotations[self.current_page_index][1]
    self.current_annotation_classifications = self.annotations[self.current_page_index][2]
    
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

    doc_names, abs_paths, doc_pages = zip(*self.page_ids)
    reviewed_dataset = []
    for doc_name, abs_path, doc_page, words, bboxs, context_images, image_bboxs, image_path, ner_tags, annotations in zip(doc_names, abs_paths, doc_pages, self.words, self.bboxs, self.context_images, self.image_bboxs, self.page_images, self.ner_tags, self.annotations):
      entry = {"doc_name": doc_name, "abs_path": abs_path, "doc_page": doc_page, "words": words, "bboxs": bboxs, "context_images":context_images, "image_bboxs":image_bboxs, "image_path":image_path, "ner_tags": ner_tags, "annotation_words":annotations[0], "annotation_idxs":annotations[1], "annotation_classifications":annotations[2]}
      reviewed_dataset.append(entry)
    
    with open(self.json_file_out, 'w') as datafile:
      json.dump(reviewed_dataset, datafile, indent=2)
      datafile.close()

    sys.stdout.write("True\n")
    sys.stdout.flush()

    self.root.destroy()

  """
  delete all annotations on the given page
  """
  def delete_all(self):
    self.current_ner_tags = {label_type: [self.ner_types[0] for _ in self.current_page_words] for label_type in self.label_types} 
    self.current_annotation_words = {label: [] for label in self.label_types}
    self.current_annotation_idxs = {label: [] for label in self.label_types} 
    self.current_annotation_classifications = {label: [] for label in self.classifications.keys()}

    self.selected_words = []
    self.selected_idxs = [] 
    self.selected_annotation_idx = None 
    self.selected_rect_ids = [] 

    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_current_box = None

    self.display_current_annotations()



def init_params():
  parser = argparse.ArgumentParser(prog="review_train.py", description="review and/or train a given model on given data")
  parser.add_argument("label_types")
  return parser.parse_args()


if __name__ == "__main__":
  params = init_params()
  label_types = ast.literal_eval(params.label_types)
  root = tk.Tk()
  app = ReviewApp(root, label_types=label_types)
  root.mainloop()