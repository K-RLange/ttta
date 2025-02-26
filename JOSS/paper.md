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
    affiliation: 3
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
 - name: ...
   index: 3
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
Text data is inherently temporal. The meaning of words and phrases changes over time, and the context in which they are used is constantly evolving. This is not just true for social media data, where the language used is rapidly influenced by current events, memes and trends, but also for journalistic, economic or political text data. Most NLP techniques however consider the corpus at hand to be homogenous in regard to time. This is a simplification that can lead to biased results, as the meaning of words and phrases can change over time. For instance, running a classic Latent Dirichlet Allocation [@bleiLatentDirichletAllocation2003] on a corpus that spans several years is not enough to capture changes in the topics over time, but only portraits an "average" topic distribution over the whole time span.

Researchers have developed a number of tools for analyzing text data over time. However, these tools are often scattered across different packages and libraries, making it difficult for researchers to use them in a consistent and reproducible way.

The `ttta` package is supposed to serve as a collection of tools for analyzing text data over time. 

# Summary
In its current state, the `ttta` package includes diachronic embeddings, dynamic topic modeling, and document scaling. These tools can be used to track changes in language use, identify emerging topics, and explore how the meaning of words and phrases has evolved over time.

Our dynamic topic model approach is based on the model RollingLDA [@RollingLDA], which is a modification of the classic Latent Dirichlet Allocation [@bleiLatentDirichletAllocation2003] that allows for the estimation of topics over time using a rolling window approach. We additionally implemented the model LDAPrototype [@riegerImprovingLatentDirichlet2020], serving as a more consistent foundation for RollingLDA than a common LDA. With these models, users can uncover and analyze topics of discussion in temporal data sets and track even rapid changes, which other dynamic topic models struggle with. This ability to track rapid changes in topics is further used in a topical change detection algorithm put forth by @TopicalChanges and @zeitenwenden that identifies change points in the word topic distribution of RollingLDA. \autoref{topical} visualizes the changes observed by the Topical Changes model in speeches from the German Bundestag, which can be analyzed further using leave-one-out word impacts provided by the model.

![Changes observed by the Topical Changes Model in a corpus of speeches held in the German Bundestag between 1949 and 2023. There is one plot for each topic, with the topic's most defining words over the time frame provided as a title for easier interpretation. Each plot shows the stability of the topic over time (blue line) as well as a threshold calculated with a monitoring procedure (orange line). A change is detected, when the blue line falls below the orange line, indicated by red vertical lines.](changes.pdf){label=topical}

The first diachronic embedding model, originally put forth by @Hamilton, uses the static word embedding model Word2Vec [@mikolovEfficientEstimationWord2013] that allows for the estimation of word embeddings over time by aligning Word2Vec vector spaces in different time chunks using a rotation matrix. The second diachronic embedding model is based on the work of @huDiachronicSenseModeling2019, who used the contextual language understanding of BERT to assign the usage of a word in a sentence to a designated sense of that word and were thus able to track the usage of a word sense over time. An example of the development of the static diachronic embedding of the word "ukraine" in the German Bundestag from 2004 to 2024 can be seen in \autoref{ukraine_plot}, which portraits the nearest neighbors of the word in the respective embedding spaces and compares them with the target word "ukraine" across the time chunks in a low dimensional plot. The plot shows the potential benefit for easily interpretable word content change, as it portraits Ukraine's context shift from being close to Russia and China, to getting closer to Europe and finally drifting into a different context, focussing on war.

![Development of the diachronic embedding of the word "ukraine" from 2004 to 2024 in the German Bundestag. Along with the word itself, its closest neighbors to visualize the target word's track across time. The dimension of the embeddings has been lowered using TSNE.](ukraine.png){label=ukraine_plot}

The Poisson Reduced Rank model [@PRR1, @PRR2] is a document scaling model, which uses a poisson-distribution based time series analysis to model the word usage of different entities (e.g. parties when analyzing party manifestos). With this model, the user will be able to visualize the difference between entities across time.

In future work, we plan to add more tools for analyzing temporal text data, as we consider the current state of the package to only be the beginning of development.

# Acknowledgements
This paper is part of a project of the Dortmund Center for data-based Media Analysis (DoCMA) at TU Dortmund University.

# References
