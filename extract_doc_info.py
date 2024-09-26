import fitz
import argparse
import json
from pathlib import Path
import sys
import ast
import subprocess
import os


NER_NULL = 'O'


# init varrious parameters and return them to the main function
# in the same order that they appear in the command line
def init_params():
  parser = argparse.ArgumentParser(prog='extract_doc_info.py',
                                   description='takes an input file and outputs a stream of text containting all the relevant category groups')
  parser.add_argument('doc_path', help='the directory containing the document to extract info from')
  parser.add_argument('json_file', help='the json file to output to')
  parser.add_argument('temp_dir', help='temporary directory where images are stored')
  parser.add_argument('label_types', help='label_types')
  parser.add_argument('file_suffix')
  parser.add_argument('-s', '--start', action='store', required=False)
  parser.add_argument('-e', '--end', action='store', required=False)
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


# return the data necessary for information extraction from the given page
def get_page_data(page):
  context_words = page.get_text("words")
  context_text = [word[4] for word in context_words]
  context_bboxs = [(word[0], word[1], word[2], word[3]) for word in context_words]
  width = page.rect.x1 - page.rect.x0
  height = page.rect.y1 - page.rect.y0
  context_bboxs = [reduce_bbox(bbox, width, height) for bbox in context_bboxs]

  return context_text, context_bboxs, width, height


# generates image for use in prediction and review (stored in tempdir)
def generate_image(page, params, page_num, doc_name):
  pix = page.get_pixmap()
  pixfile = params.temp_dir + doc_name + '_' + str(page_num) + '.png'
  pix.save(pixfile)

  return pixfile


def init_ner_tags(words, label_types):
  ner_tags = {}
  for label_type in label_types:
    ner_tags[label_type] = [NER_NULL for _ in words]
  return ner_tags


def convert_to_pdf(doc_path, output_dir):
  try:
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, doc_path], check=True)
    return True
  except subprocess.CalledProcessError as e:
    print(f"Error converting {doc_path} to PDF: {e}", file=sys.stderr)


if __name__ == "__main__":
  params = init_params()
  label_types = ast.literal_eval(params.label_types)
  suffix = params.file_suffix
  try: 
    with open(params.json_file, 'r') as file:
      existing_data = json.load(file)
  except (FileNotFoundError, json.JSONDecodeError):
    existing_data = []

  doc_path = Path(params.doc_path)

  # check if doc is word convert if true then open with pymupdf
  # ["OBJ", "REF", "TA", "HO"]
  if suffix == ".docx":
    if not convert_to_pdf(doc_path.resolve(), params.temp_dir):
      quit()
    doc_path = doc_path.with_suffix(".pdf")
    doc_path = Path(params.temp_dir, doc_path.name)
    print(os.path.exists(str(doc_path.resolve())), file=sys.stderr)
    doc = fitz.open(str(doc_path.resolve()))
  else:
    doc = fitz.open(str(doc_path.resolve()))

  start_page = int(params.start)-1 if params.start is not None else 0
  end_page = int(params.end) if params.end is not None else len(doc)
  if end_page > len(doc):
    end_page = len(doc)

  image_paths = []
  print(f"range for doc: {end_page-start_page}", file=sys.stderr)
  for i in range(end_page-start_page):
    image_paths.append(generate_image(doc[i+start_page], params, i+start_page, str(doc_path.name)))

  words_list = []
  bboxs_list = []
  width_list = []
  height_list = []
  ner_tag_list = []
  for page_num in range(start_page, end_page):
    page_words, page_bboxs, width, height = get_page_data(doc[page_num])
    words_list.append(page_words)
    bboxs_list.append(page_bboxs)
    width_list.append(width)
    height_list.append(height)
    ner_tag_list.append(init_ner_tags(page_words, label_types))
  
  new_data = [{"doc_name":doc_name, "doc_page":doc_page, "image_path":image_path, "words":words, "bboxs":bboxs, "ner_tags":ner_tags, "page_width":width, "page_height":height} for doc_name, doc_page, image_path, words, bboxs, ner_tags, width, height in zip([doc_path.name for _ in image_paths], range(start_page, end_page), image_paths, words_list, bboxs_list, ner_tag_list, width_list, height_list)]
  existing_data.extend(new_data)
  
  with open(params.json_file, 'w') as json_file:
    json.dump(existing_data, json_file, indent=2)
  
  # if needed to make the extra file then clean up as well
  if suffix == ".docx":
    os.remove(f"{params.temp_dir}{doc_path.name}")