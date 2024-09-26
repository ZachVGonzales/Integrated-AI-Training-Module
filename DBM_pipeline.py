import argparse
import sys
import os
import subprocess
import multiprocessing
from pathlib import Path
import json


# ["REV", "OBJ", "HO", "TA", "REF", "QA", "AUT"]
TEMPDIR = "tempdir/"
MODEL_TYPES = ["BIO-C", "BIO-NC"]

CONFIG_FILE = "config_file.json"
REVIEWED_DATA = "r_data.json"
UNREVIEWED_DATA = "u_data.json"
INCREMENTAL_DATA = "i_data.json"

SUPERBATCH_SIZE = 256
MINIBATCH_SIZE = 4
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def init_params():
  parser = argparse.ArgumentParser(prog='DBM_pipeline.py',
                                   description='takes an input file and outputs a stream of text containting all the relevant category groups')
  parser.add_argument('-v', '--verbose', action='store_true', required=False)
  parser.add_argument('-p', '--page_range', action='store_true', required=False)
  parser.add_argument('-o', '--ocr', action='store_true', required=False)
  return parser.parse_args()


def batch_iterator(lst, batch_size):
  for i in range(0, len(lst), batch_size):
    print(f"slice [{i}:{i+batch_size}]")
    yield lst[i:i+batch_size]


def parse_page_range(page_range: str):
  if page_range == "all" or page_range == "":
    return None
  
  parts = page_range.strip().split('-')
  try:
    if len(parts) == 2:
      start = int(parts[0].strip())
      end = int(parts[1].strip())
      return (start, end)
    else:
      raise(ValueError("Invalid page range format"))
  except Exception as e:
    print(f"Error {e}")
    return None


"""
" desc: add pdf information to collection of pdf information for this project
" params:
" - pdf_files: the new pdf files from which information will be extracted and stored in the projects datafile
" - params: the parameters initalized at program startup
"""
def init_temp_data(pdf_docx_files, params, label_types):
  for file in pdf_docx_files:
    if params.verbose:
      print(f"Processing {file}")
    
    if params.ocr and file.suffix != ".docx":
      args = [str(file.resolve()), TEMPDIR+file.name]
      if params.verbose:
        args.append("-v")
      command = [sys.executable, "ocr_doc.py"] + args
      process = subprocess.Popen(command)
      process.wait()
    
    page_range = None
    if params.page_range:
      page_range = input(f"Enter the page range for {file.name} in the format 'start_num - end_num' or 'all' for all pages:")
      page_range = parse_page_range(page_range)

    command = [sys.executable, "extract_doc_info.py", str(file.resolve()), UNREVIEWED_DATA, TEMPDIR, str(label_types), file.suffix]
    if page_range is not None:
      command.extend(["-s", str(page_range[0])])
      command.extend(["-e", str(page_range[1])])
    process = subprocess.Popen(command)
    process.wait()


"""
" DESCRIPTION: prefrom predictions on the entire existing dataset and allow user review
"""
def predict_all(label_types, model_dir):
  AI_process = subprocess.Popen([sys.executable, 'predict_train.py', model_dir, str(label_types), str(MINIBATCH_SIZE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  gui_process = subprocess.Popen([sys.executable, 'predict_gui.py', str(label_types)], stdin=subprocess.PIPE, stdout=subprocess.PIPE) # TODO: change this to specific predict gui process

  # write necessary info to AI process
  AI_process.stdin.write("predict\n".encode())
  AI_process.stdin.write(f"{UNREVIEWED_DATA}\n".encode())
  AI_process.stdin.flush()

  # wait for predictions to finish
  done = AI_process.stdout.readline().decode().strip()
  if done != "done":
    return -1

  # now that predictions are done let gui process know they are available 
  gui_process.stdin.write(f"{UNREVIEWED_DATA}\n".encode())
  gui_process.stdin.write(f"{UNREVIEWED_DATA}\n".encode())
  gui_process.stdin.flush()

  # wait for user to review predictions and GUI process will automatically write results to reviewed datafile
  done = gui_process.stdout.readline().decode().strip()
  done = bool(done)
  if done:
    return 0
  return -1


"""
" DESCRIPTION: allow the user to review all examples in the existing dataset
"              does not allow for predictions or training
"""
def review_all(label_types):
  gui_process = subprocess.Popen([sys.executable, 'review_gui.py', str(label_types)], stdin=subprocess.PIPE, stdout=subprocess.PIPE) # TODO: change this to specific predict gui process

  # have user review all data in the unreviewed datafile
  gui_process.stdin.write(f"{REVIEWED_DATA}\n".encode()) # TODO: change this back to UNREVIEWED_DATA
  gui_process.stdin.write(f"{REVIEWED_DATA}\n".encode())
  gui_process.stdin.flush()

  # wait for user to review and GUI process will automatically write results to reviewed datafile
  done = gui_process.stdout.readline().decode().strip()
  done = bool(done)
  if done:
    return 0
  return -1


"""
" DESCRIPTION: just use the existing training corpus to train the desired model
"""
def train_all(label_types, model_dir):
  AI_process = subprocess.Popen([sys.executable, 'predict_train.py', model_dir, str(label_types), str(MINIBATCH_SIZE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

  # write necessary info to AI process
  AI_process.stdin.write("train\n".encode())
  AI_process.stdin.write(f"{REVIEWED_DATA}\n".encode())
  AI_process.stdin.flush()

  # wait for training to finish
  done = AI_process.stdout.readline().decode().strip()
  if done == "done":
    return 0
  return -1


if __name__ == "__main__":
  params = init_params()
  
  # TODO: might want to change this so that data doesn't have to be extracted everytime
  # (if this line is removed the file will act as a cache that can be added to whenever) 
  #os.remove(DATAFILE) # delete the datafile if it exists already

  # run the startup window to get user input for which mode to run the program in
  command = [sys.executable, "setup_config.py", CONFIG_FILE]
  process = subprocess.Popen(command)
  process.wait()

  # also get the config so know which mode to execute
  with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)
    config_file.close()
  
  mode = config["mode"]
  label_types = config["label_types"]
  model_dir = config["model"]
  doc_dir = config["doc_dir"]
  
  os.makedirs(TEMPDIR, exist_ok=True)
  if doc_dir:
    pdf_docx_files = [file for file in Path(doc_dir).rglob("*") if file.is_file() and (file.suffix.lower() == '.pdf' or file.suffix.lower() == '.docx')]
  else:
    pdf_docx_files = []

  # ocr all docs if required and have user input desired page range if required
  # extract info from each doc in provided directory
  init_temp_data(pdf_docx_files, params, label_types)

  if mode == "predict":
    predict_all(label_types, model_dir)
  elif mode == "review":
    review_all(label_types)
  elif mode == "train":
    train_all(label_types, model_dir)
  else:
    print(f"mode {mode} is unsupported", file=sys.stderr)