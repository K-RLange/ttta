from ttta.preprocessing.semantic import OxfordDictAPI

if __name__ == '__main__':
    print(OxfordDictAPI('abuse').get_senses())
