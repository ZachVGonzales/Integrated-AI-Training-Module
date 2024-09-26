import ocrmypdf
import fitz
import argparse
import shutil
from multiprocessing import Process


# initialize command line arguments as parameters in a Namespace 
def init_params():
  parser = argparse.ArgumentParser(prog='pdf_to_text.py',
                                   description='takes an input file and outputs a stream of text containting all the relevant category groups')
  parser.add_argument('infile', help='the name of the file to ocr')
  parser.add_argument('outfile', help='the output path to store the ocrd doc at')
  parser.add_argument('-v', '--verbose', action='store_true', required=False)
  return parser.parse_args()


# start OCR on the given document 
def ocrmypdf_process(filein: str, fileout: str):
  ocrmypdf.ocr(filein, fileout, skip_text=True)


# only OCR documents that contain pages without any text
if __name__ == "__main__":
  params = init_params()
  with fitz.open(params.infile) as doc:
    for page in doc:
      if len(page.get_text("words")) == 0:
        if params.verbose:
          print("starting ocr on " + params.infile + "...")
        p = Process(target=ocrmypdf_process, args=(params.infile, params.outfile))
        p.start()
        p.join()
        quit()
    if params.verbose:
      print("copying file " + params.infile + "...")
    shutil.copyfile(params.infile, params.outfile)
    quit()