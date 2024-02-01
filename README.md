# ttta: Tools for temporal text analysis

## Project Structure
- all package-modules will be stored in src/ttta
- the modules should be separated in three subfolders: methods, preprocessing and evaluation
- preprocessing will mostly be used by the pipeline, which is independent of the individual methods
- the methods folder is designated for the models themselves. The main models should be part of the src/ttta/methods folder while auxilary models can be part of a subdirectory (e.g. the LDA folder)
- the evaluation folder is designated for potential plots and evaluation metrics 

## Desired Input Format

### Pipeline when reading from disk
I will soon write a function that creates a subdirectory with one pickle-file for each time chunk in it. The files will be named "chunk_x.pickle", with x being the number of the chunk. It will be a list of strings (for BERT-based methods) or a list of list of strings (tokenized texts for everything else) in these files. If your model is capable of processing the chunks one-at-a-time, it should contain an implementation that loops over all these files one by one as inputs.
### Pipeline when being given the data directly
The input to your models should be a pandas data frame that contains a column containing texts (preprocessed and tokenized in lists if necessary, as a plain string for BERT-based models).
In the file "preprocessing/chunk_creation.py" you can find the function "_get_time_indices". This function takes a pandas data frame (texts) with dates in the column date_column. It reorganizes the data frame into individual time chunks given an instruction in the variable "how", which can be a string (e.g. "2W" indicating that each chunk should be 2 weeks long) or a list of datetime objects. If the model is deemed to be updatable, then the variable "last_date" can be provided to check if the new chunks appear chronologically after the chunks that have already been trained by the model. It will output a new data frame detailing, at which index in the original data frame the corresponding time chunk starts in the "chunk_start"-column.
Use this function to split your data into time-chunks and then feed the texts from each individual time chunk into the function that trains your model for each individual chunk (should be able to use the same function when training from files from the disk).

## Interface for all Method-classes
Method-classes implementing NLP-models should contain the following functions:
- .save() and .load() to save/load the model itself from disk using pickle
- .fit() to fit the model
- .fit_update() if possible, to update the model with new chunk(s) after the initial training
- .get_parameters() and .set_parameters() to update/return model parameters
- .infer_vector() to infer the vector of a [CS]-token or of the words in a document. This should take an additional argument, indicating which time chunk to use the model from (e.g. does the user want the model of the first time chunk or the model of the last time chunk to infer the vector).
For an example of how to implement the first four, you can take a look at ttta/methods/rollinglda.py

## Style guide
While not being the most important part during the developement, the package should, in the end, follow one coding style. For detailed instructions, please refer to https://peps.python.org/pep-0008/. Most files can be transitioned into a PEP8 style by the autopep8 Python package. Here are some important points:
- provide docstrings for all functions, including Args and Returns (for conventions see https://peps.python.org/pep-0257/)
- provide type hints for all functions (see https://peps.python.org/pep-0484/)
- 4 spaces for indentation
- 79 characters maximum line length for code
- 72 characters maximum line length for comments/docstrings
- Avoid "if __name__ == '__main__'" if your module is not intended to be run in a top-level enviroment. For a module that only provides functions for the user or for other modules, it is unnecessary, as the module will later be called by a pipeline
- Avoid "utils" folders and files, if possible. Rename the module-names so that they are interpretable.

## What's next?
When every model is fully implemented, we need to implement unit-tests in the tests/ folder for each model. Please refer to https://docs.python.org/3/library/unittest.html for more information on unit-tests.
