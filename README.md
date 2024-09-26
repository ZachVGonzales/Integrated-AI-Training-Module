# VISION PIPELINE:

this directory contains all of the tools necessary for creating and/or training 
AI models to predict whatever classes the user may desire. The instructions for 
this can be found below. Howevwr for the purposes of the vision software 
implimentation there are certain restrictions/best practices one should follow
in order to use this program in conjunction with the postprocess program.

## USING THE DBM_PIPELINE:

The DBM_pipeline.py pythin script is the main script that should be run in 
order to perform one of the specified activities listed below.

* Review: Annotate data with the desired classes that AI shall make
  predictions on.

* Train: Train AI model(s) to be able to recognize and predict on the classes 
  within the reviewed (annotated) data.

* Predict: Have the trained model make predictions on the classes specified 
  within the startup prompt.

### COMMAND LINE FLAGS:

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

### STARTUP:

While running the DBM_pipeline.py script an initial window will be displayed
that prompts the user to input to input selections for the folling items.

* label_names: these are the different categories that AI models will trained
  to locate within documents (the actual names are unimported as how they are
  used to label thing will ultimatley dictact what predictions the model makes)
  
  NOTE: reference the section VISION RESTRICTIONS to see what labels are 
  permisbale when using the post_process program in conjunction with this one

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

### USAGE:

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

## VISION RESTRICTIONS

Since the ultimate goal of this project is to use this program to create objects
in vision out of the information extracted, there are some restrictions (or best
practices) that the user should respect in order to do this. This mostly just 
means that the user should create annotations for the following label types only.

* T_OBJ: terminal objectives
* E_OBJ: enabling objectives
* STEM: the stems of questions
* TYPE_DET: the type details of questions
* ANS: the answer to a question
* REF: items that shall linked in an xref table in vision to the document itself

TODO: add support for more vision objects

NOTE: the only way to support more label types is to add support for them into
the post_process.py program.

One other important point to take note of is the fact that some of these label
types require a secondary degree of classfication based upon the actual text 
content of that item. For example a terminal objective can be performance or
cognative. Thus for items that require this secondary layer another BERT model
is trained. The underlying workings are not that important for the user to 
understand, but it is important to note that the only what to change what 
secondary classication can be applied to what label_types is to directly change
the secondary_classifiers_config.json file.

TODO: add setup support for editing the secondary_classifiers_config.json file.

## POST PROCESSING

In order to turn the outputs of the models and revisions into actual vision
objects a layer of postprocessing is required to do this. This is because
the output of the model prediciton program is a json file and the vision
import program requires many specific files as outlined in it's documentation.

NOTE: in order to effectively use this program the input json file must be 
created by the DBM_pipeline program while abiding to the VISION RESTRICTIONS 
/ best practices.

### USAGE

In order to use this program the following comand line arguments MUST be given
in the order listed below:

* in_file: the input (.json) file that was created as the output of the
  DBM_pipeline.py program.
* out_dir: the directory that the result files of this postprocessing step
  will be created in.
* doc_dir: the original document directory or root of directory tree that
  contains within it the documents that have been processed by the DBM_pipeline

### RESULT

Once the program has been run the result of the conversion to import files will
be stored in the out_dir specified in the command line args. These files should
then be ready to be used in the vision import program and should be supported 
by version 9.14.0.0 onward (created to generate files that align with version
9.14.0.0)

### POTENTIAL ERRORS

Within the post processing step there are a few common errors that can be made
and users can prevent this by doing the following.

IMPORTANT: DO NOT extract questions from documents that are not answer keys.
This should be semi obvious, but just in case the reason is that if the answer
is not specified the program will still create a vision object, but the object
will not have an answer and will therefore be useless. This is also worsend by
the possibility of duplicate questions being created when extracting from the 
same test twice (key and not key)

Questions: Since the extraction of questions is made a bit complex by there
being so many types this is the area that the user is most likely to encounter
errors in. Any questions that are incomplete (do not have all the required 
fields) will still be created by have the missing fields set to FLAGGED.
Also questions with embedded images will have those images extracted and linked
as a document link, but the location of the image must be determined by the 
user.

Other: there are not many errors that can encountered when extracting other 
fields as they are much more straight forward, but it is still a good idea to
review the objects after they have been created.