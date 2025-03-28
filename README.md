[![PyPi](https://img.shields.io/pypi/v/ttta.svg)](https://pypi.org/project/ttta/)
[![JOSS Paper Preprint](https://img.shields.io/badge/arXiv-2503.02625-b31b1b.svg)](https://arxiv.org/abs/2503.02625)
[![Poster](https://badgen.net/badge/Poster/CPSS@Konvens24/red?icon=github)](https://github.com/K-RLange/ttta/blob/main/docs/poster.pdf)
# ttta: Tools for temporal text analysis
ttta (spoken: "triple t a") is a collection of algorithms to handle diachronic texts in an efficient and unbiased manner. 

As code for temporal text analysis papers is mostly scattered across many different repositories and varies heavily in both code quality and usage interface, we thought of a solution. ttta is designed to be a provide a collection of methods with a consistent interface and a good code quality.

**This package is currently a work in progress and in its beta stage, so there may be bugs and inconsistencies. If you encounter any, please report them in the issue tracker.**

The package is maintained and all modules were streamlined by [Kai-Robin Lange](https://lwus.statistik.tu-dortmund.de/en/chair/team/lange/).
## Contributing
If you have implemented temporal text analysis methods in Python, we would be happy to include them in this package. Your contribution will, of course, be acknowledged on this repository and all further publications. If you are interested in sharing your code, feel free to contact me at [kalange\@statistik.tu-dortmund.de](mailto:kalange@statistik.tu-dortmund.de?subject=ttta%20contribution).

## Features
- **Pipeline**: A class to help the user to use the respective methods in a consistent manner. The pipeline can be used to preprocess the data, split it into time chunks, train the model on each time chunk, and evaluate the results. The pipeline can be used to train and evaluate all methods in the package. This feature was implemented by Kai-Robin Lange. This feature is currently still work in progress and not usable.
- **Preprocessing**: Tokenization, lemmatization, stopword removal, and more. This feature was implemented by Kai-Robin Lange. 
- **LDAPrototype**: A method for more consistent LDA results by training multiple LDAs and selecting the best one - the prototype. See the [respective paper by Rieger et. al. here](https://doi.org/10.21203/rs.3.rs-1486359/v1). This feature was implemented by Kai-Robin Lange.
- **RollingLDA**: A method to train an LDA model on a time series of texts. The model is updated with each new time chunk. See the [respective paper by Rieger et. al. here](http://dx.doi.org/10.18653/v1/2021.findings-emnlp.201). This feature was implemented by Niklas Benner and Kai-Robin Lange.
- **TopicalChanges**: A method, to detect changes in word-topic distribution over time by utilizing RollingLDA and LDAPrototype and using a time-varying bootstrap control chart. See the [respective paper by Rieger et. al. here](http://ceur-ws.org/Vol-3117/paper1.pdf) and [this paper by Lange et. al. here](). This feature was implemented by Kai-Robin Lange.
- **Poisson Reduced Rank Model**: A method to train the Poisson Reduced Rank Model - a document scaling technique for temporal text data, based on a time series of term frequencies. See the [respective paper by Jentsch et. al. here](https://doi.org/10.1093/biomet/asaa063). This feature was implemented by Lars Grönberg.
- **BERT-based sense disambiguation**: A method to track the frequency of a word sense over time using BERT's contextualized embeddings. This method was inspired by the [respective paper by Hu et. al. here](https://aclanthology.org/P19-1379/). This feature was implemented by Aymane Hachcham.
- **Word2Vec-based semantic change detection**: A method that aligns Word2Vec vector spaces, trained on different time chunks, to detect changes in word meaning by comparing the embeddings. This method was inspired by [this paper by Hamilton et. al.](https://aclanthology.org/P16-1141.pdf). This feature was implemented by Imene Kolli.

## Upcoming features
- **Analyzing topical changes with the Narrative Policy Framework using LLMs**
- **Hierarchichal Sense Modeling**
- **Graphon-Network-based word sense modeling**
- **Spatiotemporal topic modeling**
- **Hopefully many more**

## Installation
You can install the latest stable release of the package from pypi. If you want the lates, unstable version, you can clone the GitHub repository.

### Using pip
```bash
pip install ttta
```

### Cloning the repository
```bash
pip install git+https://github.com/K-RLange/ttta.git
```
or
```bash
git clone https://github.com/K-RLange/ttta.git
cd ttta
pip install .
```

## Getting started
You can find a tutorial on how to use each feature of the package in the [examples folder](https://github.com/K-RLange/ttta/tree/main/examples).

## Citing ttta
If you use ttta in your research, please cite the package as follows:
```
@software{ttta,
  author = {Kai-Robin Lange, Lars Grönberg, Niklas Benner, Imene Kolli, Aymane Hachcham, Jonas Rieger and Carsten Jentsch},
  title = {ttta: Tools for temporal text analysis},
  url = {https://github.com/K-RLange/ttta},
    version = {0.9.6},
}
```
