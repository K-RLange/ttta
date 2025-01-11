---
title: 'ttta: Tools for Temporal Text Analysis'
tags:
- Python
- natural language processing
- time series
- diachronic embeddings
- dynamic topic modeling
- document scaling
authors:
  - name: Kai-Robin Lange
    affiliation: 1
  - name: Niklas Benner
    affiliation: 2
  - name: Lars Grönberg
    affiliation: 1
  - name: Aymane Hachcham
  - name: Imene Kolli
    affiliation: 4
  - name: Jonas Rieger
    affiliation: 1
  - name: Carsten Jentsch
    affiliation: 1
    
affiliations:
 - name: TU Dortmund University
   index: 1
 - name: RWI - Leibniz Institute for Economic Research
   index: 2
 - name: University of Zurich
   index: 4
citation_author: Lange et. al.
date: \today
year: 2025
bibliography: paper.bib
output: rticles::joss_article
csl: apa.csl
journal: JOSS
---

# Statement of need 
Text data is inherently temporal. The meaning of words and phrases changes over time, and the context in which they are used is constantly evolving. This is not just true for rapidly social media data, where the language used is influenced by current events, memes and trends, but also for journalistic, economic or political text data. Most NLP techniques however consider the corpus at hand to be homogenous in regard to time. This is a simplification that can lead to biased results, as the meaning of words and phrases can change over time. For instance, running a classic Latent Dirichlet Allocation \citep{bleiLatentDirichletAllocation} on a corpus that spans several years is not enough to capture changes in the topics over time, but only portraits an "average" topic distribution over the whole time span.

Researchers have developed a number of tools for analyzing text data over time. However, these tools are often scattered across different packages and libraries, making it difficult for researchers to use them in a consistent and reproducible way.

The `ttta` package is supposed to serve as a collection of tools for analyzing text data over time. 
# Summary
In its current state, the `ttta` package includes diachronic embeddings, dynamic topic modeling, and document scaling. These tools can be used to track changes in language use, identify emerging topics, and explore how the meaning of words and phrases has evolved over time. We do however only consider this to be the beginning of the development of the package. We plan to add more tools for analyzing text data over time in the future.

Our dynamic topic model approach is based on the model RollingLDA \citep{RollingLDA}, which is a modification of the classic Latent Dirichlet Allocation \citep{bleiLatentDirichletAllocation} that allows for the estimation of topics over time. We additionally implemented the model LDAPrototype \citep{riegerImprovingLatentDirichlet2020}, which is a modification of the classic LDA and serves as a foundation for the RollingLDA. RollingLDA is also the foundation for a topical change detection algorithm put forth by \cite{TopicalChanges} and \cite{Zeitenwenden} that identifies change points in the word topic distribution of RollingLDA.

The first diachronic embedding model is based on the model Diachronic Word Embeddings \citep{Hamilton}, which is a modification of the classic Word2Vec \citep{mikolovEfficientEstimationWord2013} that allows for the estimation of word embeddings over time by using a rotation matrix to align the vector spaces of the Word2Vec models. The second diachronic embedding model is based on the work of \cite{huDiachronicSenseModeling2019}, who used the contextual language understanding of BERT to assign the usage of a word in a sentence to a designated sense of that word, according to the Oxford dictionary, and were thus able to track the usage of a word sense over time.

The Poisson Reduced Rank model \citep{PRR1, PRR2} is a document scaling model, which uses a poisson-distribution based time series analysis to model the word usage of different entities (e.g. parties when analyzing party manifestos). With this model, the user will be able to visualize the difference between entities across time.

# Acknowledgements
This paper is part of a project of the Dortmund Center for data-based Media Analysis (DoCMA) at TU Dortmund University.
# References
