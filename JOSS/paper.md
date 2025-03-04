---
title: 'ttta: Tools for Temporal Text Analysis'
tags:
- Python
- natural language processing
- time series
- dynamic topic modeling
- change detection
- diachronic embeddings
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
    affiliation: 3
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
   index: 3
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

Our dynamic topic model approach is based on the model RollingLDA [@RollingLDA], which is a modification of the classic Latent Dirichlet Allocation [@bleiLatentDirichletAllocation2003], that allows for the estimation of topics over time using a rolling window approach. We additionally implemented the model LDAPrototype [@riegerImprovingLatentDirichlet2020], serving as a more consistent foundation for RollingLDA than a common LDA. With these models, users can uncover and analyze topics of discussion in temporal data sets and track even rapid changes, which other dynamic topic models struggle with. This ability to track rapid changes in topics is further used in the Topical Changes model put forth by @TopicalChanges and @zeitenwenden that identifies change points in the word topic distribution of RollingLDA. \autoref{fig:topical} visualizes the changes observed by the Topical Changes model in speeches from the German Bundestag [@SpeakGer], which can be analyzed further using leave-one-out word impacts provided by the model or, as @NarrativeShiftDetection proposed, by asking Large Language Models to interpret the change and relate it to a possible narrative shift.

![Changes observed by the Topical Changes Model in a corpus of speeches held in the German Bundestag between 1949 and 2023. There is one plot for each topic, with the topic's most defining words over the time frame provided as a title for easier interpretation. Each plot shows the stability of the topic over time (blue line) as well as a threshold calculated with a monitoring procedure (orange line). A change is detected, when the observed stability falls below the threshold, indicated by red vertical lines.\label{fig:topical}](changes.pdf)

The first diachronic embedding model, originally introduced by @Hamilton, builds on the static word embedding model Word2Vec [@mikolovEfficientEstimationWord2013]. It enables the estimation of word embeddings over time by aligning Word2Vec vector spaces across different time chunks using a rotation matrix. The second diachronic embedding model is based on the work of @huDiachronicSenseModeling2019, who leveraged BERT's contextual language understanding to associate word usage in a sentence with a specific word sense, thus enabling users to track shifts in word meanings over time.

An example of the evolution of the static diachronic embedding of the word Ukraine in the German Bundestag from 2004 to 2024 is shown in \autoref{fig:ukraineplot}. The plot shows the nearest neighbors of Ukraine in the respective embedding spaces, enabling users to observe the trajectory of the target word Ukraine across time chunks in a low-dimensional representation. This visualization highlights the potential of using diachronic embeddings for the interpretation and detection of word context change, as it reflects Ukraine's contextual shift from being closely associated with Russia and China, to aligning more with Europe, and ultimately moving into a different context centered on war.

![Development of the diachronic embedding of the word "ukraine" from 2004 to 2024 in the German Bundestag. Along with the word itself, its closest neighbors to visualize the target word's track across time. The dimension of the embeddings has been lowered using TSNE.\label{fig:ukraineplot}](ukraine.png)

The Poisson Reduced Rank model [@PRR1; @PRR2] is a document scaling model, which uses a poisson-distribution based time series analysis to model the word usage of different entities (e.g. parties when analyzing party manifestos). With this model, the user are able to analyze, how the entities move in a latent space that is generated using the word usage counts.

In future work, we plan to add more tools for analyzing temporal text data, as we consider the current state of the package to only be the beginning of development.

# Acknowledgements
This paper is part of a project of the Dortmund Center for data-based Media Analysis (DoCMA) at TU Dortmund University.

# References
