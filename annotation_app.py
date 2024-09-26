import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import sys
import fitz # only used in AnnotationApp (Maybe seperate into own files)
import ast # only used in ReviewApp
import queue
import threading


"""
TODO: should probably make a super class for three different subclasses
"""

class AnnotationApp():
  """
  params:
  - root: the tkinter Tk object that controls the window
  - page_lim: the limit on the number of pages that shall be extracted from a given document,
  -           defaults to 10
  """
  def __init__(self, root: tk.Tk, page_lim: int = 10, label_types: list = ["Default"], json_out: str = "annotations.json", classifications: dict = {"T_OBJ": ["performance", "cognative"], "E_OBJ": ["performance", "cognative"], "STEM": ["MC", "TF", "SA", "ES"]}) -> None:
    self.root = root
    self.json_out = json_out
    self.page_lim = page_lim
    self.root.title = "Annotation App"
    
    # create a list of label types and NER types given the input list of labels
    self.label_types = label_types 
    self.classifications = classifications
    self.ner_types = ["O"]
    for label in label_types:
      self.ner_types.append(f"B-{label}")
      self.ner_types.append(f"I-{label}")
    self.label2id = {l:i for i, l in enumerate(self.label_types)}
    self.id2nertype = {i:t for i, t in enumerate(self.ner_types)}
    self.nertype2id = {t:i for i, t in enumerate(self.ner_types)}

    """Internal Variables storing info"""

    # info just used by the app for all loaded pages, does not need to be saved
    self.page_images = [] # a list of image information for each page

    # for viewing the currently created labels and their types:
    self.annotations = [] # list of tuples of dicts where for each dict, key = "label type", value = [words0, words1, ...] where wordsN is a list of words for each annotation

    #NOTE: need to change but this is the info that MUST be output to the json in a list of dicts
    self.page_ids = [] # list of tuples, NOTE: must be seperated into diferent colums before storing
    self.words = [] # list of lists, each is a list of the words on a given page (just the text)
    self.bboxs = [] # list of lists, each is a list of the bounding boxes on a given page
    self.context_images = []
    self.image_bboxs = []
    self.ner_tags = [] # list of dicts, each as long as the context words 

    # info pertaining to the current page being examined in the app
    self.current_page_index = 0 
    self.current_page_words = [] # just the text of the words on the current page [word0, word1, ...]
    self.current_page_bboxs = [] # list of tuples for each bbox on current page, [(x0, y0, x1, y1), ...]
    self.current_ner_tags = [] # list of dicts of lists of NER tags (one for each word)
    self.current_annotation_words = {} # dict: key = "label type", value = [words0, words1, ...] where wordsN is a list of words for each annotation
    self.current_annotation_idxs = {} # dict: key = "label type", value = [idxs0, idxs1, ...] 
    self.current_annotation_classifications = None
    self.photo = None # a reference to the current photo being displayed on the canvas (so as not to be deleted by garbage collection)

    # info pertaining to the current annotation being made
    self.selected_words = [] # just the text of the words selected (to display to user)
    self.selected_idxs = [] # a list of the indexs for the selected text in the list of all the words on the page
    self.selected_annotation_idx = None # the int of the annotation in the current annoation list in dict, if it exists
    self.selected_rect_ids = [] # a list of the ids of the rectangle objects currently being displayed on the canvas, so they can be deleted later

    """UI Components"""
    # frame and canvases for image / annotations
    self.top_frame = tk.Frame(root)
    self.top_frame.pack(fill='both', expand=True)

    self.left_canvas = tk.Canvas(self.top_frame, bg="white", width=50, height=250)
    self.left_canvas.pack(side='left', fill='both', expand=False)

    self.right_canvas = tk.Canvas(self.top_frame, bg="lightgrey", width=150, height=250)
    self.right_canvas.pack(side='right', fill='both', expand=True)

    self.bottom_frame = tk.Frame(root)
    self.bottom_frame.pack(side='bottom', fill='x')
    
    # Create Button Components
    self.btn_load = tk.Button(self.bottom_frame, text="Load Files", command=self.load_pages)
    self.btn_load.pack(side=tk.LEFT)

    self.btn_save = tk.Button(self.bottom_frame, text="Save Annotation", command=self.save_annotation)
    self.btn_save.pack(side=tk.LEFT)

    self.btn_prev = tk.Button(self.bottom_frame, text="Previous Image", command=self.prev_page)
    self.btn_prev.pack(side=tk.LEFT)

    self.btn_next = tk.Button(self.bottom_frame, text="Next Image", command=self.next_page)
    self.btn_next.pack(side=tk.LEFT)

    self.btn_del_annot = tk.Button(self.bottom_frame, text="Delete Annotation", command=self.delete_annotation)
    self.btn_del_annot.pack(side=tk.LEFT)

    self.btn_del = tk.Button(self.bottom_frame, text="Delete Page", command=self.del_page)
    self.btn_del.pack(side=tk.LEFT)

    self.btn_close = tk.Button(self.bottom_frame, text="Save/Close", command=self.save_and_close)
    self.btn_close.pack(side=tk.LEFT)

    self.selected_label = tk.StringVar(self.bottom_frame, self.label_types[0])
    self.label_menu = tk.OptionMenu(self.bottom_frame, self.selected_label, *self.label_types, command=self.update_selected_label)
    self.label_menu.pack(side='top')

    # Create Annotation viewing portion:
    self.annotation_scrollbar = tk.Scrollbar(self.right_canvas, orient="vertical", command=self.right_canvas.yview)
    self.annotation_scrollbar.pack(side='right', fill='y')
    self.right_canvas.config(yscrollcommand=self.annotation_scrollbar.set)

    self.select_start_x = None
    self.select_start_y = None
    self.select_current_box = None

    self.deselect_start_x = None
    self.deselect_start_y = None
    self.deselect_current_box = None

    # for displaying the text that has been selected to user
    self.selected_text = tk.Label(self.root, text="", anchor="w", justify="left", bg="white", fg="black")
    self.selected_text.pack(side='bottom')

    self.left_canvas.bind("<ButtonPress-1>", self.select_down) # on left mouse down
    self.left_canvas.bind("<B1-Motion>", self.select_drag) # on left mouse motion while down
    self.left_canvas.bind("<ButtonRelease-1>", self.select_release) # on left mouse release
    self.left_canvas.bind("<ButtonPress-3>", self.deselect_down) # on right mouse down
    self.left_canvas.bind("<B3-Motion>", self.deselect_drag) # on right mouse drag while down
    self.left_canvas.bind("<ButtonRelease-3>", self.deselect_release) # on right mouse release

  """
  load images and image words from user provided pdf documents using pymupdf
  """
  def load_pages(self):
    file_paths = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF files", "*.pdf")], initialdir="./")
    for path in file_paths: 
      with fitz.open(path) as doc:
        page_range = self.page_lim if (self.page_lim and self.page_lim<len(doc)) else len(doc)
        for i in range(page_range):
          page = doc[i]
          page_words = page.get_text("words")
          self.page_images.append(page.get_pixmap())
          self.annotations.append(({label: [] for label in self.label_types}, {label: [] for label in self.label_types}, {label: [] for label in self.classifications.keys()}))
          self.page_ids.append((os.path.basename(path), i))
          self.words.append([word[4] for word in page_words])
          self.bboxs.append([(word[0], word[1], word[2], word[3]) for word in page_words])
          self.ner_tags.append({label_type: [self.ner_types[0] for _ in page_words] for label_type in self.label_types}) # init list of ner_tags for each page
    
    self.load_page() # load the first page to start editing
  
  """
  load the image at the current index into view for the user
  """
  def load_page(self):
    # if the image list is empty don't do anything
    if not self.page_images:
      return
    
    image = self.page_images[self.current_page_index]
    image = Image.frombytes("RGB", (image.width, image.height), image.samples)
    
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

  def save_page(self):
    if not self.page_images:
      return 
    
    self.words[self.current_page_index] = self.current_page_words
    self.bboxs[self.current_page_index] = self.current_page_bboxs
    self.ner_tags[self.current_page_index] = self.current_ner_tags
    self.annotations[self.current_page_index] = (self.current_annotation_words, self.current_annotation_idxs, self.current_annotation_classifications)

  """
  go to the next page in the page list
  """
  def next_page(self):
    if not self.page_images:
      return
    
    self.save_page() # save page before changing
    if self.current_page_index < (len(self.page_images)-1):
      self.current_page_index += 1
      self.load_page()
  
  """
  go to the previous page in the page list
  """
  def prev_page(self):
    if not self.page_images:
      return

    self.save_page() # save page before changing
    if self.current_page_index > 0:
      self.current_page_index -= 1
      self.load_page()
  
  """
  delete the current page in the list
  """
  def del_page(self):
    if not self.page_images:
      return

    del self.page_images[self.current_page_index]
    del self.page_ids[self.current_page_index]
    del self.words[self.current_page_index]
    del self.bboxs[self.current_page_index]
    del self.context_images[self.current_page_index]
    del self.image_bboxs[self.current_page_index]
    del self.ner_tags[self.current_page_index]
    del self.annotations[self.current_page_index]
    
    if not self.page_images:
      return
    if self.current_page_index >= len(self.page_images):
      self.current_page_index -= 1
    self.load_page()

  """
  delete the current selection box showing and create a new one
  """
  def select_down(self, event):
    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_current_box = None

    self.select_start_x = event.x
    self.select_start_y = event.y
    self.select_current_box = self.left_canvas.create_rectangle(self.select_start_x, self.select_start_y, self.select_start_x, self.select_start_y, outline="blue")

  """
  expand the selection box do new mouse location
  """
  def select_drag(self, event):
    self.left_canvas.coords(self.select_current_box, self.select_start_x, self.select_start_y, event.x, event.y)

  """
  save the select bounding box and process
  """
  def select_release(self, event):
    end_x, end_y = event.x, event.y
    self.process_selection(self.select_start_x, self.select_start_y, end_x, end_y)
    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_current_box = None
  
  """
  delete the current deselection box showing and create a new one
  """
  def deselect_down(self, event):
    if self.deselect_current_box:
      self.left_canvas.delete(self.deselect_current_box)
      self.deselect_current_box = None
    
    self.deselect_start_x = event.x
    self.deselect_start_y = event.y
    self.deselect_current_box = self.left_canvas.create_rectangle(self.deselect_start_x, self.deselect_start_y, self.deselect_start_x, self.deselect_start_y, outline="red")

  """
  expand the deselection box to new mouse location
  """
  def deselect_drag(self, event):
    self.left_canvas.coords(self.deselect_current_box, self.deselect_start_x, self.deselect_start_y, event.x, event.y)

  """
  save the deselect bounding box and process
  """
  def deselect_release(self, event):
    end_x, end_y = event.x, event.y
    self.process_deselection(self.deselect_start_x, self.deselect_start_y, end_x, end_y)
    if self.deselect_current_box:
      self.left_canvas.delete(self.deselect_current_box)
      self.deselect_current_box = None

  """
  take a given box and save all words/indexs in selected cache if encompased by the given box
  """
  def process_selection(self, x0, y0, x1, y1):
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    selected_words = [] # the words that have been selected by user
    selected_idxs = [] # the indexs of said words within the total word list
    for idx, (text, box) in enumerate(zip(self.current_page_words, self.current_page_bboxs)):
      if idx not in self.selected_idxs and box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1:
        selected_words.append(text)
        selected_idxs.append(idx)

    self.selected_words += selected_words
    self.selected_idxs += selected_idxs
    self.selected_text.config(text=" ".join(self.selected_words))
    #print(f"selected words: {self.selected_words}\nselected idxs: {self.selected_idxs}")
    self.update_selected_bboxs()

  """
  take a given box and remove any words/indexs in selected cache if they are encompased by the given box
  """
  def process_deselection(self, x0, y0, x1, y1):
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    new_selected_words = []
    new_selected_idxs = []
    for idx, word in zip(self.selected_idxs, self.selected_words):
      box = self.current_page_bboxs[idx]
      if box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1:
        # if deselected dont add to new lists
        continue
      new_selected_idxs.append(idx)
      new_selected_words.append(word)
    
    self.selected_words = new_selected_words
    self.selected_idxs = new_selected_idxs
    self.selected_text.config(text=" ".join(self.selected_words))
    self.update_selected_bboxs()

  """
  update a label to the new selection
  """
  def update_selected_label(self, selection):
    self.selected_label.set(selection)

  """
  save the current selection of words/indexs for the annotation into the associated lists/dicts
  """
  def save_annotation(self):
    selected_label = self.selected_label.get()
    selected_label_id = self.label2id[selected_label]

    if self.selected_idxs:
      b_idx = self.selected_idxs[0]
      self.current_ner_tags[selected_label][b_idx] = self.id2nertype[selected_label_id*2 +1]

    for idx in self.selected_idxs[1:]:
      self.current_ner_tags[selected_label][idx] = self.id2nertype[selected_label_id*2 +2]
    
    if self.selected_annotation_idx is not None:
      self.current_annotation_words[self.selected_label.get()][self.selected_annotation_idx] = self.selected_words
      self.current_annotation_idxs[self.selected_label.get()][self.selected_annotation_idx] = self.selected_idxs
    else:
      self.current_annotation_words[self.selected_label.get()].append(self.selected_words)
      self.current_annotation_idxs[self.selected_label.get()].append(self.selected_idxs)
      if selected_label in self.classifications.keys():
        self.current_annotation_classifications[selected_label].append(self.classifications[selected_label][0])
    
    #print(self.current_annotation_words)
    #print(self.current_annotation_idxs)
    #print(self.current_ner_tags)
    self.display_current_annotations()
    
    # delete the bounding box and remove stored info (also deslect)
    self.selected_words = []
    self.selected_idxs = []
    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_start_x = None
      self.select_start_y = None
      self.select_current_box = None
    if self.deselect_current_box:
      self.left_canvas.delete(self.deselect_current_box)
      self.deselect_start_x = None
      self.deselect_start_y = None
      self.deselect_current_box = None
    self.selected_annotation_idx = None
    self.selected_text.config(text="")
    self.update_selected_bboxs()

  """
  load the given annotation into the selected cache given the correct index and label
  """
  def select_annotation(self, idx, label):
    self.selected_label.set(label)
    self.selected_words = self.current_annotation_words[label][idx]
    self.selected_idxs = self.current_annotation_idxs[label][idx]
    self.selected_text.config(text=" ".join(self.selected_words))
    self.selected_annotation_idx = idx
    self.update_selected_bboxs()

  """
  unload any information currently stored in the selected cache
  """
  def deselect_annotation(self):
    if self.select_current_box:
      self.left_canvas.delete(self.select_current_box)
      self.select_current_box = None
    self.selected_words = []
    self.selected_idxs = []
    self.selected_text.config(text="")
    self.selected_annotation_idx = None
    self.update_selected_bboxs()

  """
  delete the selected annotation if one exists and deselect any info in the selected cache
  """
  def delete_annotation(self):
    if self.selected_annotation_idx is not None:
      selected_label = self.selected_label.get()
      for idx in self.current_annotation_idxs[selected_label][self.selected_annotation_idx]:
        self.current_ner_tags[selected_label][idx] = self.id2nertype[0] # change the tags that are being deleted to null
      del self.current_annotation_words[selected_label][self.selected_annotation_idx]
      del self.current_annotation_idxs[selected_label][self.selected_annotation_idx]
    self.deselect_annotation()
    self.display_current_annotations()

  """
  updates the right display of labels and associated annotations
  """
  def display_current_annotations(self):
    self.right_canvas.delete("all")

    text_frame = tk.Frame(self.right_canvas)
    self.right_canvas.create_window((0, 0), window=text_frame, anchor="nw")

    for label, annotations in self.current_annotation_words.items():
      # Create and pack the label
      key_label = tk.Label(text_frame, text=label, font=("Arial", 12, "bold"))
      key_label.pack(anchor='nw', padx=5, pady=(5, 0))
        
      # Create and pack the text widget
      for i, annotation in enumerate(annotations):
        # create a subframe for each annotaion
        annotation_frame = tk.Frame(text_frame)
        annotation_frame.pack(fill=tk.X, padx=5, pady=2)

        text_widget = tk.Text(annotation_frame, wrap=tk.WORD, height=4, width=50)
        text_widget.pack(side='left', padx=5, pady=(0, 10))
        text_widget.insert(tk.END, annotation)
        text_widget.config(state=tk.DISABLED)  # Make text widget read-only

        select_button = tk.Button(annotation_frame, text="Select", command=lambda idx=i, label=label: self.select_annotation(idx, label))
        select_button.pack(side='right', padx=5)

        delete_button = tk.Button(annotation_frame, text="Delete", command=lambda idx=i, label=label: (self.select_annotation(idx, label), self.delete_annotation()))
        delete_button.pack(side="right", padx=5)

        if label in self.classifications.keys():
          selected_class = tk.StringVar(annotation_frame, self.current_annotation_classifications[label][i])
          class_menu = tk.OptionMenu(annotation_frame, selected_class, *self.classifications[label], command=lambda selection, idx=i, label=label, string_var=selected_class: self.update_selected_class(string_var, idx, label, selection))
          class_menu.pack(side='right', padx=5)

    # Update canvas scroll region
    text_frame.update_idletasks()
    self.right_canvas.config(scrollregion=self.right_canvas.bbox("all"))
  
  def update_selected_class(self, string_var: tk.StringVar, idx: int, label: str, selection: str):
    string_var.set(selection)
    self.current_annotation_classifications[label][idx] = selection

  """
  display the bounding boxes of the currently selected annotation
  """
  def update_selected_bboxs(self):
    self.remove_selected_bboxs()
    for idx in self.selected_idxs:
      x0, y0, x1, y1 = self.current_page_bboxs[idx]
      self.selected_rect_ids.append(self.left_canvas.create_rectangle(x0, y0, x1, y1, outline="blue"))

  """
  removes the bouding boxes of the selected annotation from the canvas
  """
  def remove_selected_bboxs(self):
    while len(self.selected_rect_ids) > 0:
      id = self.selected_rect_ids.pop(0)
      self.left_canvas.delete(id)

  """
  dumps all anotations saved into a json file, formatted for ease of use with huggingface datasets
  """
  def save_and_close(self):
    self.save_page()

    doc_names, doc_pages = zip(*self.page_ids)
    output_list = []
    for doc_name, doc_page, words, bboxs, ner_tags in zip(doc_names, doc_pages, self.words, self.bboxs, self.ner_tags):
      output_list.append({"doc_name": doc_name, "doc_page": doc_page, "words": words, "bboxs": bboxs, "ner_tags": ner_tags})

    with open(self.json_out, 'w') as json_file:
      json.dump(output_list, json_file, indent=2)

    self.root.destroy()








class ReviewApp(AnnotationApp):
  """
  params:
  - root: the tkinter Tk object that controls the window
  """
  def __init__(self, root: tk.Tk, label_types: list = ["Default"], json_out: str = "annotations.json") -> None:
    super().__init__(root=root, label_types=label_types, json_out=json_out)
    self.input_queue = queue.Queue()
    self.thread = threading.Thread(target=self.read_data)
    self.thread.daemon = True
    self.thread.start()
    self.export_ready = False

    # extra review specific UI buttons
    self.btn_del_all = tk.Button(self.bottom_frame, text="DEL ALL", command=self.delete_all)
    self.btn_del_all.pack(side=tk.LEFT)

    self.btn_export = tk.Button(self.bottom_frame, text="Export Batch", command=self.export_batch)
    self.btn_export.pack(side=tk.LEFT)

  """
  read data continuously from parent (this is run in seperate thread)
  """
  def read_data(self):
    while True:
      line = sys.stdin.readline().strip()
      if line == '':
        break
      self.input_queue.put(line)

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
    batch = self.input_queue.get_nowait()
    if batch:
      batch = json.loads(batch)
      for data in batch:
        doc_name = data["doc_name"]
        page_num = data["doc_page"]
        image_path = data["image_path"]
        words = data["words"]
        bboxs = data["bboxs"]
        ner_tags = data["word_tags"] # dict where key = label_type, value = ner_tags for given type

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
    self.current_annotation_classifications = self.annotations[self.current_page_index][2]
    self.photo = ImageTk.PhotoImage(image) # save a reference to the photo object

    self.left_canvas.config(width=image.width, height=image.height) # change shape to match photo
    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo) # add image to canvas
    self.display_current_annotations()

  """
  export the current batch now that it has been reviewed
  """
  def export_batch(self):
    if not self.export_ready:
      return

    self.save_page()

    doc_names, doc_pages = zip(*self.page_ids)
    reviewed_batch = []
    reviewed_batch_annotations = []
    for doc_name, doc_page, words, bboxs, image_path, ner_tags, annotations in zip(doc_names, doc_pages, self.words, self.bboxs, self.page_images, self.ner_tags, self.annotations):
      reviewed_batch.append({"doc_name": doc_name, "doc_page": doc_page, "words": words, "bboxs": bboxs, "image_path":image_path, "word_tags": ner_tags})
      
      annotations_words = annotations[0]
      reviewed_batch_annotations.append({"doc_name":doc_name, "doc_page":doc_page})
      for label, annotations in annotations_words.items():
        reviewed_batch_annotations[-1][label] = annotations
    
    reviewed_batch = json.dumps(reviewed_batch) + "\n"
    sys.stdout.write(reviewed_batch)
    sys.stdout.flush()
    reviewed_batch_annotations = json.dumps(reviewed_batch_annotations) + "\n"
    sys.stdout.write(reviewed_batch_annotations)
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


if __name__ == "__main__":
  root = tk.Tk()
  app = ReviewApp(root, label_types=["OBJ", "OE", "TR", "EM", "REV"])
  root.mainloop()