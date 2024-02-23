from utils.utils import *




class DiachronicEmbeddings():
    def __init__(self, **kwargs) -> None:
        pass


    @classmethod
    def from_word2vec(cls, **kwargs):
        """
        Initialize a DiachronicEmbeddings object from a word2vec model.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DiachronicEmbeddings: A DiachronicEmbeddings object.
        """
        return cls(**kwargs)




    def get_model(self, model_name):
        if model_name == 'Word2Vec':
            from word2vec.word2vec import Word2VecModel
            return Word2VecModel()
        
        else:
            raise ValueError('model unknown')
        


    
    def train(self, corpus):
        """
        Train the model.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def load(self, path):
        """
        Load the model.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def predict(self, **kwargs):
        """
        Predict the embeddings.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

            

    
        
    
    
        
if __name__ == '__main__':
    w2v = DiachronicEmbeddings.from_word2vec()
 
    w2v.train()
    w2v.load()
    w2v.predict()
