from gensim.models import Word2Vec
from utils.utils import *

class Word2VecModel():

    def __init__(self, 
                min_count = 5,
                window = 5,
                negative = 5,
                ns_exponent = 0.75,
                vector_size = 300,
                sg = 1,
                workers = 1,
                **kwargs):
        self.word2vec_model = Word2Vec(
            min_count = min_count,
            window = window,
            negative = negative,
            ns_exponent = ns_exponent,
            vector_size = vector_size,
            sg = sg,
            workers = workers,
            **kwargs
        )

    
    def train(self, 
              corpus: list, 
              epochs = 10, 
              start_alpha = 0.025, 
              end_alpha = 0.0001,
              compute_loss = True,
              **kwargs):
    

        self.word2vec_model.build_vocab(corpus)
        total_examples = self.word2vec_model.corpus_count

        self.word2vec_model.train(
            corpus,
            total_examples=total_examples,
            epochs= epochs,
            start_alpha= start_alpha,
            end_alpha= end_alpha,
            compute_loss= compute_loss,
            **kwargs
        )
        
        return self.word2vec_model
    

    def align(self, reference_model, **kwargs):
        """
        Align the model.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """

        try:
            if kwargs['method'] == 'procrustes':
                return reference_model, smart_procrustes_align_gensim(reference_model.word2vec_model,self.word2vec_model)
            
            else:
                raise NotImplementedError('Only procrustes method is implemented')
        
        except KeyError:
            raise ValueError('method must be specified')
        
    
    def save(self, path):
        """
        Save the model.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.word2vec_model.save(path)
            

    
    # def pipeline(self, **kwargs):
    #     config_path = kwargs.get('config', 'word2vec.yaml')
    #     config = read_yaml(config_path)['Word2Vec']

        
    #     training_config = config['pipeline']['training']
    #     alignment_config = config['pipeline']['alignment']
        
        


    #     periods = kwargs.get('periods', None)
    #     if periods is None:
    #         raise ValueError('periods must be specified')
    #     elif not isinstance(periods, list):
    #         raise ValueError('periods must be a list of strings')
        
    #     train = kwargs.get('train', False)
    #     align = kwargs.get('align', False)
        

    #     if train:
    #         training_output_dir = kwargs.get('training_output_dir', None)
    #         # f'{directory}/output/word2vec/{trial_name}/train'

    #         if training_output_dir is None:
    #             raise ValueError('training_output_dir must be specified')
    #         elif not isinstance(training_output_dir, str):
    #             raise ValueError('training_output_dir must be string')
            
    #         if not os.path.exists(training_output_dir):
    #             os.makedirs(training_output_dir)
            
    #         print('Training Word2Vec models. Saving to: ', training_output_dir)

    #         corpus = kwargs.get('corpus', None)
    #         if corpus is None:
    #             raise ValueError('Training requires corpus as input')
            
    #         elif not isinstance(corpus, list):
    #             raise ValueError('corpus must list of lists of strings')
            
    #         if len(corpus) != len(periods):
    #             raise ValueError('corpus must be same length as periods')
            
            

    #         for period_number, period in enumerate(periods):
    #             word2vec_model = Word2Vec(**training_config['model_params'])
    #             self.build_vocab(word2vec_model, corpus[period_number])
    #             self.train(word2vec_model, corpus[period_number], **training_config['training_params'])
    #             self.save(word2vec_model, path= training_output_dir + f'/raw_{period}.model')

    #         print('Training complete.')
        


    #     if align:
    #         alignment_output_dir = kwargs.get('alignment_output_dir', None)
    #         # f'{directory}/output/word2vec/{trial_name}/align'

    #         if alignment_output_dir is None:
    #             raise ValueError('alignment_output_dir must be specified')
    #         elif not isinstance(alignment_output_dir, str):
    #             raise ValueError('alignment_output_dir must be string')
            
    #         if not os.path.exists(alignment_output_dir):
    #             os.makedirs(alignment_output_dir)
            
            

    #         all_models_path = kwargs.get('alignment_models_path', None)
    #         if all_models_path is None:
    #             raise ValueError('alignment_models_path must be specified')
    #         elif not isinstance(all_models_path, str):
    #             raise ValueError('alignment_models_path must be string')
            

        
    #         all_models = [self.load(model) for model in sorted(glob(all_models_path + '/*.model'))]

    #         reference_model = all_models[alignment_config['reference']]
    #         reference_period = periods[alignment_config['reference']]
    #         self.save(reference_model, path= alignment_output_dir + f'/aligned_{reference_period}.model')

    #         alignment_models = all_models[:alignment_config['reference']] + all_models[alignment_config['reference'] + 1:]
    #         alignment_periods = periods[:alignment_config['reference']] + periods[alignment_config['reference'] + 1:]

    #         for model, period in zip(alignment_models, alignment_periods):
    #             if alignment_config['method'] == 'orthogonal_procrustes':
    #                 aligned_model = smart_procrustes_align_gensim(reference_model,model)
    #             else:
    #                 raise ValueError('alignment method not supported')
    #             self.save(aligned_model, path= alignment_output_dir + f'/aligned_{period}.model')

    #     return training_output_dir if train else None, alignment_output_dir if align else None


        



        


    

if __name__ == '__main__':
    w2v = Word2VecModel(window = 10)

    # corpus_2015 = read_txt('../../../../input/1980_articles.txt')[:10]
    # w2v = Word2VecModel()
    # w2v.train(corpus_2015, size= 100, window= 5, min_count= 1, workers= 4, sg= 0, iter= 5)
    # w2v.save(w2v, 'output/word2vec/trial_1/train/raw_1980.model')
    # model = w2v.load('output/word2vec/trial_1/train/raw_1980.model')
    # print(model.wv['the'])
    # print(model.wv.most_similar('the'))
    # print(model.wv.similarity('the', 'then'))



    # _, _ = w2v.pipeline(periods = ['1980'], train= True, align = False, training_output_dir = '../../../../output/word2vec/trial_1/train', corpus = [corpus_2015])

    # _, _ = w2v.pipeline(periods = ['1980', '2015'], 
    #                     train= True,
    #                     training_output_dir = '../../../../output/word2vec/trial_1/train',
    #                     corpus = [corpus_2015, corpus_2015],
    #                     align = True,
    #                     alignment_output_dir = '../../../../output/word2vec/trial_1/align',
    #                     alignment_models_path = '../../../../output/word2vec/trial_1/train',
    #                     )
   
    




        

    

        
        
