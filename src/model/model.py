import os
from collections import defaultdict
import argparse
from tqdm.auto import trange
import numpy as np
import pandas as pd
from util import (safelog, 
                  normalize, 
                  lex_to_str)

class BaseRSA:
    def __init__(self, contexts, costs, alpha, prior):
        self.contexts = contexts
        self.costs = costs
        self.alpha = alpha
        self.prior = prior
 
    def L_0(self, A, context=None):
        ''' Literal listener: normalizes the defined lexicon '''
        if context is None:
            return normalize(A * self.prior)
        else:
            return normalize(A * self.contexts[context] * self.prior)

    def S_p(self, A, context=None):
        ''' Pragmatic speaker: softmax over the defined lexicon taking costs into account '''
        if context is None:
            return normalize(np.exp(self.alpha * (safelog(self.L_0(A, None).T) + self.costs)))
        else:
            return normalize(np.exp(self.alpha * (safelog(self.L_0(A, context).T) + self.costs)))

    def L_p(self, A, context=None):
        ''' Pragmatic listener: normalizes the defined lexicon receiver from the pragmatic speaker '''
        if context is None:
            return normalize(self.S_p(A, None).T * self.prior)
        else:
            return normalize(self.S_p(A, context).T * self.contexts[context] * self.prior)
        
class Agent(BaseRSA):
    def __init__(self, n_words, n_meanings, contexts, costs, alpha, prior):
        BaseRSA.__init__(self, contexts, costs, alpha, prior)
        self.lexicons = self.generate_lexicons(n_words, n_meanings)
        self.lexicons_norm = np.array([normalize(lexicon) for lexicon in self.lexicons]) 
        self.lexicon_probs = np.ones(len(self.lexicons)) / len(self.lexicons)

    @staticmethod
    def generate_lexicons(n_words, n_meanings):
        ''' Generate all possible productive lexicons (excluding option with null singals) '''
        arrays =  np.array([list(map(int, list(np.binary_repr(i, width=n_words*n_meanings)))) 
                            for i in range(2**(n_words*n_meanings))])
        lexicons = arrays.reshape((2**(n_words*n_meanings), n_words, n_meanings))
        return lexicons[lexicons.sum(axis=1).min(axis=1) > 0]

    def speaker(self, meaning, context):
        lexicon = self.lexicons[np.random.choice(len(self.lexicons), p=self.lexicon_probs)]
        return self.S_p(lexicon, context)[meaning].argmax()
    
    def listener(self, message, context):
        lexicon = self.lexicons[np.random.choice(len(self.lexicons), p=self.lexicon_probs)]
        return self.L_p(lexicon, context)[message].argmax()
    
    def update_lexicon(self, message, meaning):
        ''' 
        Get the posterior distribution over lexicons given the message and meaning

        To-do:

        - Update on L_p instead of L_0?

        NB: adding random noise to prevent agents getting stuck in local optima 
        '''
        self.lexicon_probs = normalize(self.lexicon_probs * self.lexicons_norm[:, message, meaning] + 1e-5)
    
    def get_lexicon(self):
        ''' Get the lexicon with the highest probability '''
        return self.lexicons[self.lexicon_probs.argmax()], self.lexicon_probs.argmax()
    
    def L_p_max(self):
        return self.L_p(self.lexicons[self.lexicon_probs.argmax()], None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--costs', type=str)
    args = parser.parse_args()

    # cost will be a list of costs for each meaning, converted to a numpy array
    args.costs = np.array(args.costs.split(',')).astype(float)

    context_meanings = {0: [0, 1], 1: [0], 2: [1]}
    CONTEXTS = np.array([[1, 1, 0], [1, 0, 1]])
    N_MEANINGS = 3
    N_WORDS = 2
    PRIOR = normalize(np.array([1, 1, 1]))

    logs_comp = []

    for tr in trange(args.n_iter, 
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                    desc='Running the model', 
                    position=0, 
                    leave=True):
        agent1 = Agent(n_words = N_WORDS, 
                       n_meanings = N_MEANINGS, 
                       contexts = CONTEXTS, 
                       costs = args.costs, 
                       alpha = args.alpha, 
                       prior = PRIOR)
        agent2 = Agent(n_words = N_WORDS, 
                       n_meanings = N_MEANINGS, 
                       contexts = CONTEXTS, 
                       costs = args.costs, 
                       alpha = args.alpha, 
                       prior = PRIOR)
        
        logs = defaultdict(dict)

        for _ in range(100):
            meaning = np.random.randint(0, N_MEANINGS)
            context = np.random.choice(context_meanings[meaning])
            logs[_] = {'trial': tr, 'context': context, 'meaning': meaning}
            if _ % 2 == 0:
                message = agent1.speaker(meaning, context)
                meaning_guess = agent2.listener(message, context)
                logs[_]['message'] = message
                logs[_]['meaning_guess'] = meaning_guess
                if meaning_guess == meaning:
                    logs[_]['correct'] = True
                    agent1.update_lexicon(message, meaning)
                    agent2.update_lexicon(message, meaning)
                else:
                    logs[_]['correct'] = False
            else:
                message = agent2.speaker(meaning, context)
                meaning_guess = agent1.listener(message, context)  
                logs[_]['message'] = message
                logs[_]['meaning_guess'] = meaning_guess
                if meaning_guess == meaning:
                    logs[_]['correct'] = True
                    agent1.update_lexicon(message, meaning)
                    agent2.update_lexicon(message, meaning)
                else:
                    logs[_]['correct'] = False
            logs[_]['lexicon_1'] = lex_to_str(agent1.get_lexicon()[0])
            logs[_]['lexicon_2'] = lex_to_str(agent2.get_lexicon()[0])
        logs_comp.append(logs)

    logs_comb = pd.concat([pd.DataFrame(k).T.reset_index() for k in logs_comp]).reset_index(drop=True)
    # if no folder "logs" exists in the "src" folder, create one in the same folder
    if not os.path.exists('src/logs/'):
        os.makedirs('src/logs/')
    
    logs_comb.to_csv(f'src/logs/logs-{args.n_iter}-{args.alpha}.csv', index=False)
    