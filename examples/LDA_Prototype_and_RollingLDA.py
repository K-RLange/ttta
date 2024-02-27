import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from ttta.methods.lda_prototype import LDAPrototype
from ttta.methods.rolling_lda import RollingLDA
from ttta.methods.topical_changes import TopicalChanges
from ttta.preprocessing.preprocess import preprocess

# Preprocessing for toy examples
df = pd.DataFrame(twitter_samples.docs())[["text", "created_at"]]
df["created_at"] = pd.to_datetime(df["created_at"])
df["text"] = preprocess(df["text"])
# The texts must be preprocessed into a list of lists of words. The preprocess-function can be replaced by any
# arbitrary preprocessing function that includes tokenization.
# ----------------------------LDAPrototype----------------------------
# print(help(LDAPrototype))
# Most options can be set in the __init__ method. Only options that can vary from fit to fit,
# such as epochs, are set as parameters in fit.
# Here, we train an LDA with 30 topics, 5 LDAs to be compared for LDAPrototype, and a
# minimum threshold of 5 (absolute) and 0.002 (relative) for topic matching in the prototype step.
lda = LDAPrototype(10, prototype=10, topic_threshold=[3, 0.001], verbose=1)
# verbose can be set to 0 for no output or 2 for more output
# lda.fit(df["text"], epochs=300)
# print(lda.top_words(10))

# -----------------------------RollingLDA-----------------------------
# print(help(RollingLDA))
# RollingLDA initializes an LDAPrototype object and continuously trains it on different time chunks. The most important RollingLDA
# parameters, such as memory or warmup (currently only whole numbers specifying the number of time chunks), are set in __init__.
# In __init__, a distinction is made between "initial_epochs" and "subsequent_epochs" if the initial
# fit should have more epochs. Another important parameter is "how", which specifies the type of separation of the data into time chunks.
# This can either be a list of datetime objects or a string indicating the chunk intervals. "2W" stands for 2 weeks and 30Min for 30 Minutes,
# for example. If there are too few data in a chunk (less than min_docs_per_chunk, default 10 * K), then chunks are merged.
# The fit method expects a pandas DataFrame as input, whose relevant text and date columns can be customized.

roll = RollingLDA(10, prototype=10, topic_threshold=[2, 0.001], initial_epochs=300, subsequent_epochs=150, memory=2,
                  warmup=3, how="30Min")
roll.fit(df, text_column="text", date_column="created_at")
# The top words from individual chunks can be obtained...
print(roll.top_words(0))
print(roll.top_words(2))
# or combined from all chunks.
print(roll.top_words())

# The models also yield other methods, such as wordclouds, which allows you to plot word clouds for a chunk
# or save them to a PDF.
roll.wordclouds(chunks=[1], path="wordclouds")
# This can also be done for all chunks at once, but it takes time...
# roll.wordclouds(path="wordclouds")

# If new data needs to be fitted, fit_update should be called, where "how" can be adjusted for the new
# data.
# If you want to change the interval size to monthly intervals, you can do it like this:
# However, this is not executed here because the model throws an error if the data of the new texts overlap with
# those of the old texts.
# roll.fit_update(df, how="M")

# A test of the Topical Changes paper is also implemented, i.e., a change detection on individual topics.
# The theoretical topic changes over time are plotted against the realized topic changes.
# A vertical red line indicates a change point.
# Interpretation is done via leave-one-out word impacts (whether they are calculated correctly still needs to be
# verified, the results are unusually uninformative). Alternatively, the top words of the topics for the
# respective time chunks can be used.
changes = TopicalChanges(roll, mixture=0.8, reference_period=2, samples=300)
changes.plot_distances()
print(changes.word_impact())

# Both models can be easily saved
roll.save("roll.pickle")
lda.save("lda.pickle")
# and loaded again
roll = RollingLDA(10)
roll.load("roll.pickle")
lda = LDAPrototype(10)
lda.load("lda.pickle")

# For more information, call the help pages for each method via
# help(TopicalChanges)
# and if you are still unsure about something, find bugs, or have feature requests, feel free to open an issue on Github or
# write me a mail: kalange@statistik.tu-dortmund.de :)