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
  parser.add_argument('-c', '--clean', action='store_true', required=False)
  parser.add_argument('-p', '--page_range', action='store_true', required=False)
  parser.add_argument('-r', '--revision', action='store_true', required=False)
  parser.add_argument('-o', '--ocr', action='store_true', required=False)
  parser.add_argument('-t', '--train', action='store_true', required=False)
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
" description: use the information stored within the given project datafile to incrementally review and train the AI
" parameters:
" - 
"""
def incremental(extracted_data, label_types, model_dir):

  gui_process = subprocess.Popen([sys.executable, 'incremental_gui.py', str(label_types)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  AI_process = subprocess.Popen([sys.executable, 'predict_train.py', model_dir, str(label_types), str(MINIBATCH_SIZE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

  print(f"total num pages {len(extracted_data)}")
  for super_batch in batch_iterator(extracted_data, SUPERBATCH_SIZE):
    # get info from gui on if AI predictions shall be made on the batch
    predict = gui_process.stdout.readline().decode().strip()
    predict = bool(predict)

    # dump the super_batch data to the temporary file
    with open(INCREMENTAL_DATA, 'w') as temp_df:
      json.dump(super_batch, temp_df, indent=2)
      temp_df.close()

    # if predictions shall be made then output request to the AI process
    # otherwise just let the gui process know that the data is available
    # for review
    if predict:
      AI_process.stdin.write("predict\n".encode())
      AI_process.stdin.write(f"{INCREMENTAL_DATA}\n".encode())
      AI_process.stdin.flush()
    
      # wait for predictions to be writenback then make these available to gui process
      done = AI_process.stdout.readline().decode().strip()
      if done == "done":
        print(done)
        gui_process.stdin.write(f"{INCREMENTAL_DATA}\n".encode())
        gui_process.stdin.flush()
      else:
        break
    else:
      gui_process.stdin.write(f"{INCREMENTAL_DATA}\n".encode())
      gui_process.stdin.flush()
    
    # wait for gui process to review the predictions made by the AI process
    train = gui_process.stdout.readline().decode().strip()
    train = bool(train)

    # now that batch is reviewed add it to the training corpus
    with open(REVIEWED_DATA, 'r+') as training_df:
      all_training_data = json.load(training_df)
      reviewed_data = []
      with open(INCREMENTAL_DATA, 'r') as temp_df:
        reviewed_data = json.load(temp_df)
        temp_df.close()
      all_training_data.extend(reviewed_data)
      json.dump(all_training_data, training_df)
    
    # now check if gui process requested training, if yes then train on whole training corpus
    if train:
      AI_process.stdin.write("train\n".encode())
      AI_process.stdin.write(f"{REVIEWED_DATA}\n".encode())
      AI_process.stdin.flush()

      # wait for the training of the model to finish
      done = AI_process.stdout.readline().decode().strip()
      if done != "done":
        break
  
  # once done with all batches then quit
  print("done processing projects extracted data incrimentally")
  quit()


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
  gui_process.stdin.write(f"{REVIEWED_DATA}\n".encode())
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
  gui_process.stdin.write(f"{UNREVIEWED_DATA}\n".encode())
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

  # now that all the required info is gathered from the documents review a batch and train (mandatory for 1st optional for next)
  with open(UNREVIEWED_DATA, 'r') as datafile:
    extracted_data = json.load(datafile)
    datafile.close()

  if mode == "incremental":
    incremental(extracted_data, label_types, model_dir)
  elif mode == "predict":
    predict_all(label_types, model_dir)
  elif mode == "review":
    review_all(label_types)
  elif mode == "train":
    train_all(label_types, model_dir)
  else:
    print(f"mode {mode} is unsupported", file=sys.stderr)