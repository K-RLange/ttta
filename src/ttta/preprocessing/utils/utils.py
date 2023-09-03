import xmltodict
import os
from glob import glob
import re
import yaml




    

def read_yaml(file_path: str) -> dict:
    """
    Read in a yaml file and return a dictionary.

    Args:
        file_path (str): The path to the yaml file.

    Returns:
        dict: The dictionary.
    """
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    

def read_xml(file_path: str) -> dict:
    """
    Read in an xml file and return a dictionary.

    Args:
        file_path (str): The path to the xml file.

    Returns:
        dict: The dictionary.
    """
    
    with open(file_path, 'r') as f:
        xml_string = f.read()
    return xmltodict.parse(xml_string)



def get_records(file_path: str) -> list:
    """
    Read in the xml file and return a list of articles.

    Args:
        file_path (str): The path to the xml file.

    Returns:
        list: A list of articles.
    """
    xml_dict = read_xml(file_path)
    return xml_dict['records']['record']


def get_article_text(article: dict) -> str:
    """
    Get the text from an article.

    Args:
        article (dict): The article.

    Returns:
        str: The text of the article.
    """
    if 'fulltext' not in article.keys():
        return ''

    else:
        return article['fulltext']

 

def get_articles(file_path: str) -> list:
    """
    Get all articles from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list of articles.
    """

    records = get_records(file_path)
    articles = []
    for record in records:
        article = get_article_text(record)

        if type(article) == list:
            article = ' '.join(article)

        if article is None:
            continue
        
        else:
            articles.append(article)
    
    return articles



def save_texts(texts: list, file_path: str):
    """
    Save a list of sentences to a file.

    Args:
        sentences (list): A list of sentences.
        file_path (str): The path to the file.
    """
    with open(file_path, 'w') as f:
        for line in texts:
            f.write(line + '\n')



# ------------------- cleantxt --------------------
def cleantxt(text, options: dict):
    
    newtext = re.sub('\n', ' ', text) # Remove ordinary linebreaks (there shouldn't be, so this might be redundant)

    if options['remove_punctuation']:
        newtext = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', str(newtext)) # Remove anything that is not a space, a letter, a dot, or a number
    if options['lowercase']:
        newtext = str(newtext).lower() # Lowercase
    
    if options['lemmatize']:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet 

        lemmatizer = WordNetLemmatizer()

        newtext = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x, wordnet.VERB), newtext.split())))
    
    if options['remove_stopwords']:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        newtext = ' '.join(list(filter(lambda x: x not in stop_words, newtext.split())))

    return newtext










if __name__ == '__main__':
    path = '../../data/articles_raw_data/'
    file_path = glob(os.path.join(path, '*.xml'))[0]

    print(len(read_xml(file_path)['records']['record']))
    