from utils.utils import *
from glob import glob




class DiachronicEmbeddings():
    def __init__(self, model) -> None:
        self.model = model


    @classmethod
    def from_word2vec(cls, **kwargs):
        """
        Initialize a DiachronicEmbeddings object from a word2vec model.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DiachronicEmbeddings: A DiachronicEmbeddings object.
        """

        from word2vec.word2vec import Word2VecModel
        return cls(model = Word2VecModel(**kwargs))

        

    
    def train(self, corpora: list, align: bool = True, ref: int = -1, **kwargs) -> None:
        """
        Train the model.

        Args:
            corpora: A list of lists (each one is a corpus).
            **kwargs: Arbitrary keyword arguments.

        """
        raw = []
        for corpus in corpora:
            m = self.model
            m.train(corpus, **kwargs)
            raw.append(m)
            del m

        print('len raw', len(raw))
        
        if align:
            aligned = []
            reference_model = raw[ref]
            for i in range(len(raw)):
                print(i)
                if i == ref:
                    aligned.append(reference_model)
                    print('skipped')
                    continue
                new = raw[i].align(reference_model = reference_model, method = 'procrustes')
                aligned.append(new)
                del new

        
        return {'raw': raw, 'aligned': aligned, 'reference_model': reference_model}
    

        

    def load(self, model_dir: str, **kwargs):
        """
        Load the model.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """

        paths = sorted(glob(model_dir + '/*.model'))
        models = [self.model.load(path) for path in paths]
        



    def predict(self, **kwargs):
        """
        Predict the embeddings.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass



    # def load(self, path):
    #     return Word2Vec.load(path)
    
    def save(self, model, path):
        model.save(path)

    
    # def get_embedding(self, model, word):
    #     return model.wv[word]
    
    # def most_similar(self, model, word):
    #     return model.wv.most_similar(word)
    
    # def similarity(self, model, word1, word2):
    #     return model.wv.similarity(word1, word2)
            

    
        
    
    
        
if __name__ == '__main__':
    corpus_1980 = read_txt('../../../input/1980_articles.txt')[:10]
    corpus_1990 = read_txt('../../../input/1990_articles.txt')[:10]
    corpus_2000 = read_txt('../../../input/2000_articles.txt')[:10]
    corpus_2010 = read_txt('../../../input/2010_articles.txt')[:10]

    corpora = [corpus_1980, corpus_1990, corpus_2000, corpus_2010]
    print('loaded corpora')


    w2v = DiachronicEmbeddings.from_word2vec(window = 10)


    print('initialized model')
    print('training models')
    
    models = w2v.train(corpora=corpora, align = True, epochs = 2)
    print('trained models')

    


    w2v.save(models['reference_model'], 'reference_model.model')
    print('saved reference model')

    model = w2v.load('reference_model.model')



    
    
    




    
