# DBM PIPELINE:

The DBM Pipeline is a tool for annotating documents of the pdf or word type
and then using said annotations to train an AI Model (LayoutLMv3) for NER
information extraction.

## FUNCTIONALITY:

The DBM_pipeline.py pythin script is the main script that should be run in 
order to perform one of the specified activities listed below.

* Review: Annotate data with the desired classes that AI shall make
  predictions on.

* Train: Train AI model(s) to be able to recognize and predict on the classes 
  within the reviewed (annotated) data.

* Predict: Have the trained model make predictions on the classes specified 
  within the startup prompt.

## COMMAND LINE FLAGS:

these are mostly unimportant beside providing some additionaly user control for 
how information shall be extracted.

* -o: if set the program will attempt to perform OCR on any documents that need
  it. This can be extremely useful (necessary) when dealinng with pdfs that are 
  just rastorized images and do not contain any text information

* -p: if set the user will be prompted to select the page range that shall have
  information extracted from it. This is mostly obsolete since the user can 
  delete pages while reviewing data and it's not really a good idea to remove 
  data from documents in general (as it can lead to worse model performance).
  However this can be useful when dealing with a few extremely large documents
  in order to lessen the extraction time / storage load.

* -v: doesn't do too much right now, program is a little more vocal.
  TODO: make verbose talk more

## STARTUP:

While running the DBM_pipeline.py script an initial window will be displayed
that prompts the user to input to input selections for the folling items.

* label_names: these are the different categories that AI models will trained
  to locate within documents (the actual names are unimported as how they are
  used to label thing will ultimatley dictact what predictions the model makes)

* mode: this is selected from one of the above listed activites

* model: this is the directory that stores the model(s) that have either been 
  trained or will be trained.  If no model(s) yet exist that is ok, just 
  select the directory where they shall be stored.

* doc_dir: this is the directory or root of directory tree containing documents
  from which information shall be extracted.

  NOTE: the doc_dir should only be specified once when the review mode is ran
  to create training data and the only once more when the predict mode is ran 
  to add the rest of the documents that shall be predicted upon. This is because
  the information is "cached" by the program in a file called "u_data.json" and
  that file will only be appended to, not overwritten

  TODO: this is were functionality is likely to change as the program is updated
  since data control/flow should be placed more in the users hands.

## USAGE:

Once the program has been started it will preform different functions given the
different mode selected:

* REVIEW:
  this will launch a window in which any information in the "u_data.json" file 
  will be displayed to the user along with options for annotating the 
  information contained within each document page.  The possible annotations 
  that can be made are determined by the user's choices for labels. specifiing 
  a model or doc_dir is NOT required for this mode, however it is recomended to
  specify a doc_dir if there is new information that needs annotation or if the 
  "u_data.json" file does not yet exist.

* TRAIN:
  this will not launch any window. Instead the user can view the progress of 
  the training within the comfort of their command prompt. This program simply 
  uses the reviewed data (data in the "r_data.json" file) to train AI model(s) 
  to be able to preform the annotations themselves. So the required parameter 
  for this mode is the model. If no models currently exist that is ok, just 
  specify an empty directory where they should be stored.  If interested you 
  can learn moreabout the training method / model type in the section 
  ADDITIONAL INFO.

* PREDICT:
  At first no window will be launched as predictions for all label_types
  specified must be made. Once this is done the predictions that the model has
  made will be displayed in the same kind of window as the review mode. Once 
  the user has reviewed the predictions made by the model to be accurate the 
  user can export the data to the "r_data.json" file.

## Dependencies:

* Python 3.0+
* Pillow (image manipulation)
* PyMuPDF (pdf data extraction)
* HuggingFace transformers (LayoutLMv3 specifically) and datasets
* Pytorch
* Soffice (optional -> .docx support)
* Ocrmypdf (optional -> ocr flag support)
* nltk (optional -> synonym replacement support)