from semantic_shift.data.data_loader import Loader
from semantic_shift.data.data_preprocessing import PREPROCESS

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
        

    return corpora         
        
        




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
        max_documents = 10000,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        )
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(r, f, indent=4)











