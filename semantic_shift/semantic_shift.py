from semantic_shift.data.data_loader import Loader
from semantic_shift.data.data_preprocessing import PREPROCESS
from semantic_shift.embeddings.word2vec import Word2VecTrainer, WordEmbeddings as w2v
from semantic_shift.embeddings.roberta import RobertaTrainer, WordEmbeddings as roberta, MaskedWordInference
from semantic_shift.embeddings.bert import BertTrainer, WordEmbeddings as bert

import json
import os


def main(output_dir, data_path, periods, **kwargs):

    print('*'*10, 'Loading data', '*'*10, '\n')
    # file_type = data_path.split('.')[-1]
    corpora = {}
    # if file_type == 'xml':
    xml_tag = kwargs['xml_tag']
    for period in periods:
        path = data_path.format(period)
        corpora[period] = Loader.from_xml(
            path, 
            xml_tag
            ).forward(
                target_words=kwargs['target_words'], 
                max_documents=kwargs['max_documents'], 
                shuffle=kwargs['shuffle']
                )
        # corpora[period] = Loader.from_txt(path).forward()
        print('Found {} documents in corpus: {}'.format(len(corpora[period]), period))




        print('*'*10, 'Preprocessing data', '*'*10, '\n')
        corpora[period] = list(map(lambda x: PREPROCESS().forward(x, **kwargs['preprocessing_options']), corpora[period]))
        print('Finished preprocessing')




        print('*'*10, 'Feature extraction: Training Embeddings', '*'*10, '\n')
        # training
        if kwargs['diachronic_embeddings_options']['model_type'] == 'word2vec':
            trainer = Word2VecTrainer(
                **kwargs['diachronic_embeddings_options']['model_options']
            )

            trainer.train(
                corpora[period], 
                output_dir=f"{output_dir}/{period}"
                )

        elif kwargs['diachronic_embeddings_options']['model_type'] == 'roberta':
            trainer = RobertaTrainer(
                **kwargs['diachronic_embeddings_options']['model_options']
            )

            trainer.train(
                corpora[period], 
                output_dir=f"{output_dir}/{period}"
                )

        elif kwargs['diachronic_embeddings_options']['model_type'] == 'bert':
            trainer = BertTrainer(
                **kwargs['diachronic_embeddings_options']['model_options']
            )

            trainer.train(
                corpora[period], 
                words_to_mask=kwargs['target_words'],
                output_dir=f"{output_dir}/{period}"
                )

        else:
            raise NotImplementedError('Model type not implemented')
        
        print('Finished training embeddings')




        print('*'*10, 'Feature extraction: Infering embeddings', '*'*10, '\n')
        # infering
        if kwargs['diachronic_embeddings_options']['model_type'] == 'word2vec':
            embeddings = w2v(
                **kwargs['diachronic_embeddings_options']['model_options']['model_path']
            )
        
        elif kwargs['diachronic_embeddings_options']['model_type'] == 'roberta':
            embeddings = roberta(
                **kwargs['diachronic_embeddings_options']['model_options']['model_path']
            )
        
        elif kwargs['diachronic_embeddings_options']['model_type'] == 'bert':
            embeddings = bert(
                **kwargs['diachronic_embeddings_options']['model_options']['model_path']
            )
        
        else:
            raise NotImplementedError('Model type not implemented')
        
        word_embedding = {}

        for word in kwargs['target_words']:
            if kwargs['diachronic_embeddings_options']['model_type'] == 'word2vec':
                word_embedding[word] = embeddings.infer_vector(word)
            
            else:
                word_embedding[word] = []
                for doc in corpora[period]:
                    word_embedding[word].append(embeddings.infer_vector(word, doc))
                 

    return word_embedding         
        
        




if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    periods = [
        1980,
        # 1982,
        # 1985,
        # 1987,
        # 1989,
        # 1990,
        # 1992,
        # 1995,
        # 2000,
        # 2001,
        # 2005,
        # 2008,
        # 2010,
        # 2012,
        # 2015,
        # 2017
    ]

    xml_tag = 'fulltext'
    file_path = "input/xml/TheNewYorkTimes{}.xml"

    preprocessing_options = {
        "remove_stopwords": False, 
        "remove_punctuation": True, 
        "remove_numbers": False, 
        "lowercase": True, 
        "lemmatize": True
        }

    target_words = [
            "office",
            "gay",
            "abuse",
            "king",
            "apple",
            "bank",
            "war",
            "love",
            "money",
            "school",
            "police",
            "family",
            "work"
        ]
    
    r = main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_words = target_words,
        max_documents = 10,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        )
    
    diachronic_embeddings_options = {
        "model_type": "word2vec",
        "model_options": {
            "window": 15,
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(r, f, indent=4)











