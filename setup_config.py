import tkinter as tk
from tkinter import filedialog
import json
import argparse
import ast


class StartupWindow():
  """
  """
  def __init__(self, root: tk.Tk, config_file) -> None:
    self.root = root
    self.config_file = config_file
    
    self.root.title = "Startup Config"
    root.geometry("400x200")

    # entry for label_types:
    self.l_label = tk.Label(root, text="enter label types, format: type1, type2, ...")
    self.l_label.pack(pady=5)
    self.label_types_entry = tk.Entry(root)
    self.label_types_entry.pack(pady=5)

    # selection for mode of program:
    self.mode_label = tk.Label(root, text="select mode:")
    self.mode_label.pack(pady=5)
    self.options = ["incremental", "predict", "review", "train"]
    self.selected_option = tk.StringVar()
    self.selected_option.set(self.options[0])

    self.dropdown = tk.OptionMenu(root, self.selected_option, *self.options)
    self.dropdown.pack(pady=5)

    # directory selector for model
    self.selected_model = tk.StringVar()
    self.select_model_btn = tk.Button(root, text="Select Model", command=self.open_model_dialog)
    self.select_model_btn.pack(pady=5)

    # directory selector for doc_dir
    self.doc_dir = tk.StringVar()
    self.doc_dir_btn = tk.Button(root, text="Select doc dir", command=self.open_dir_dialog)
    self.doc_dir_btn.pack(pady=5)

    # save and quit bttn:
    self.submit = tk.Button(root, text="save config and quit", command=self.save_config_and_quit)
    self.submit.pack(pady=10)

  def open_model_dialog(self):
    directory = filedialog.askdirectory()
    if directory:
      self.selected_model.set(directory)

  def open_dir_dialog(self):
    directory = filedialog.askdirectory()
    if directory:
      self.doc_dir.set(directory)

  def save_config_and_quit(self):
    label_types = self.label_types_entry.get()
    mode = self.selected_option.get()
    model = self.selected_model.get()
    doc_dir = self.doc_dir.get()
    config_output = {"label_types": ast.literal_eval(label_types), "mode":mode, "model":model, "doc_dir":doc_dir}
    with open(self.config_file, "w") as cf:
      json.dump(config_output, cf)
    self.root.quit()


def init_params():
  parser = argparse.ArgumentParser(prog='DBM_pipeline.py',
                                   description='takes an input file and outputs a stream of text containting all the relevant category groups')
  parser.add_argument('config_file', help='the file to which the configuration will be output')
  return parser.parse_args()


if __name__ == "__main__":
  params = init_params()
  confing_file = params.config_file
  root = tk.Tk()
  setup = StartupWindow(root, confing_file)
  root.mainloop()