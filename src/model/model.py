import os
from collections import defaultdict
import argparse
from tqdm.auto import trange
import numpy as np
import pandas as pd
from util import (generate_lexicons, 
                  lex_to_str)
from scipy.stats import entropy

np.seterr(divide='ignore', invalid='ignore')

NOISE = 1e-10


class BaseRSA:
    def __init__(self, alpha, prior, C, contexts):
        self.contexts = contexts
        self.alpha = alpha
        self.prior = prior
        self.C = C

    @staticmethod
    def safelog(vals):
        with np.errstate(divide='ignore'):
            return np.log(vals)

    @staticmethod
    def normalize(vals):
        return np.nan_to_num(vals / np.sum(vals, axis=-1, keepdims=True))
    
    def L_0(self, L, c):
        if c is None:
            return self.normalize(L * self.prior)
        return self.normalize(L * self.prior * self.contexts[c])
    
    def S_0(self, L, c):
        if c is None:
            return self.normalize(L.transpose(0, 2, 1) )
        return self.normalize(np.exp(self.safelog(L.transpose(0, 2, 1)) - self.C))
    
    def S_p(self, L, c=None):
        if c is None:
            return self.normalize(np.exp(self.alpha * (self.safelog(self.L_0(L, c).transpose(0, 2, 1)) - self.C)))
        return self.normalize(np.exp(self.alpha * (self.safelog(self.L_0(L, c).transpose(0, 2, 1)) - self.C)))
    
    def L_p(self, L, c=None):
        if c is None:
            return self.normalize(self.S_p(L, c).transpose(0, 2, 1) * self.prior)
        return self.normalize((self.S_p(L, c).transpose(0, 2, 1) * self.prior * self.contexts[c])) 

class Agent(BaseRSA):
    def __init__(self, alpha, prior, C, n_words, n_meanings, contexts, model="RSA"):
        super().__init__(alpha, prior, C, contexts)
        self.Lexicons = generate_lexicons(n_words, n_meanings)
        self.prob_lexicon = np.ones(len(self.Lexicons)) / len(self.Lexicons)
        self.n_words = n_words
        self.n_meanings = n_meanings
        self.model = model
    
    def speaker(self, m, c):
        # index of the lexicon with the highest probability given prob_lexicon
        if self.model == "RSA":
            lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
            # return self.S_p(self.Lexicons, c)[lexicon_idx][m].argmax()
            return np.random.choice(np.arange(self.n_words), p=self.S_p(self.Lexicons, c)[lexicon_idx][m])
        elif self.model == "naive":
            lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
            return np.random.choice(np.arange(self.n_words), p=self.S_0(self.Lexicons, c)[lexicon_idx][m])
    
    def listener(self, w, c):
        # index of the lexicon with the highest probability given prob_lexicon
        if self.model == "RSA":
            lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
            # return self.L_p(self.Lexicons, c)[lexicon_idx][w].argmax()
            if np.sum(self.L_p(self.Lexicons, c)[lexicon_idx][w]) < 1:
                return np.random.choice(np.arange(self.n_meanings), p=[1 / self.n_meanings] * self.n_meanings)
            else:
                return np.random.choice(np.arange(self.n_meanings), p=self.L_p(self.Lexicons, c)[lexicon_idx][w])
        elif self.model == "naive":
            lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
            if np.sum(self.L_0(self.Lexicons, c)[lexicon_idx][w]) < 1:
                return np.random.choice(np.arange(self.n_meanings), p=[1 / self.n_meanings] * self.n_meanings)
            else:
                return np.random.choice(np.arange(self.n_meanings), p=self.L_0(self.Lexicons, c)[lexicon_idx][w])

    
    def update(self, w, m, c, correct, role):
        if role == "speaker":
            if self.model == "RSA":
                if correct:
                    self.prob_lexicon = self.normalize(self.S_p(self.Lexicons, c)[:, m, w] * self.prob_lexicon + NOISE)
                else:
                    self.prob_lexicon = self.normalize(self.S_p(self.Lexicons, c)[:, m, 1 - w] * self.prob_lexicon + NOISE)
            elif self.model == "naive":
                if correct:
                    self.prob_lexicon = self.normalize(self.S_0(self.Lexicons, c)[:, m, w] * self.prob_lexicon + NOISE)
                else:
                    self.prob_lexicon = self.normalize(self.S_0(self.Lexicons, c)[:, m, 1 - w] * self.prob_lexicon + NOISE)
        elif role == "listener":
            if self.model == "RSA":
                if correct:
                    self.prob_lexicon = self.normalize(self.L_p(self.Lexicons, c)[:, w, m] 
                    * self.prob_lexicon + NOISE)
                else:
                    meanings = range(self.n_meanings)
                    for meaning in meanings:
                        if meaning != m:
                            self.prob_lexicon = self.normalize(self.L_p(self.Lexicons, c)[:, w, meaning]
                            * self.prob_lexicon + NOISE)
            elif self.model == "naive":
                if correct:
                    self.prob_lexicon = self.normalize(self.L_0(self.Lexicons, c)[:, w, m] 
                    * self.prob_lexicon + NOISE)
                else:
                    meanings = range(self.n_meanings)
                    for meaning in meanings:
                        if meaning != m:
                            self.prob_lexicon = self.normalize(self.L_0(self.Lexicons, c)[:, w, meaning]
                            * self.prob_lexicon + NOISE)

    def max_prob_L(self):
        return lex_to_str(self.Lexicons[np.argmax(self.prob_lexicon)])
    

class Experiment:
    def __init__(self, alpha, prior, C, contexts, n_iter, n_rounds, model):
        self.n_iter = n_iter
        self.n_rounds = n_rounds

        self.logs = defaultdict(lambda: defaultdict(dict))
        self.group_posterior = []

        self.alpha = alpha
        self.prior = prior
        self.C = C
        self.contexts = contexts
        self.model = model
        

    def one_round(self, a, b, m, c, i, r):
        w = a.speaker(m, c)
        g = b.listener(w, c)
        # if m == g:
        a.update(w, m, c, m == g, "speaker")
        b.update(w, m, c, m == g, "listener")
        self.logs[i][r]['word'] = w
        self.logs[i][r]['word_length'] = self.C[w]
        self.logs[i][r]['guess'] = g
        self.logs[i][r]['correct'] = 1 if (m == g) else 0
        self.logs[i][r]['lexicon_a'] = a.max_prob_L()
        self.logs[i][r]['lexicon_b'] = b.max_prob_L()
        self.logs[i][r]['entropy_a'] = entropy(a.prob_lexicon)
        self.logs[i][r]['entropy_b'] = entropy(a.prob_lexicon)
    
    def run(self):
        for i in range(self.n_iter):
            self.agents = [Agent(alpha=self.alpha, prior=self.prior, C=self.C, n_words=2, n_meanings=3, contexts=self.contexts, model=self.model),
                            Agent(alpha=self.alpha, prior=self.prior, C=self.C, n_words=2, n_meanings=3, contexts=self.contexts, model=self.model)]
            for r in range(self.n_rounds):
                m = np.random.choice([0, 1, 2], p=self.prior)
                # if meaning is 0, context is either 0 or 1, if 1, context is 0, if 2 context is 1
                c = np.random.choice([0, 1]) if m == 0 else m - 1
                self.logs[i][r]['meaning'] = m
                self.logs[i][r]['context'] = c
                if r % 2 == 0:
                    self.one_round(self.agents[0], 
                                    self.agents[1], 
                                    m, c, i, r)
                else:
                    self.one_round(self.agents[1], 
                                    self.agents[0], 
                                    m, c, i, r)
            
            self.group_posterior.append(self.agents[0].prob_lexicon)
            self.group_posterior.append(self.agents[1].prob_lexicon)
                        
    def save(self, rd=False):
        df = pd.DataFrame.from_dict({(i, r): self.logs[i][r]
                                        for i in self.logs.keys()
                                        for r in self.logs[i].keys()},
                                        orient='index').reset_index()
        df.columns = ['trial', 'round', 'meaning', 'context', 'word', 'word_length', 'guess', 'correct', 'lexicon_a', 'lexicon_b', 'entropy_a', 'entropy_b']
        if not rd:
            df.to_csv(f'src/data/logs/logs-{self.n_iter}-{int(self.alpha)}.csv', index=False)
        else:
            return df
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--costs', type=str)
    parser.add_argument('--priors', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    args.model = "RSA" if args.model == "rsa" else "naive"
    args.costs = np.array(args.costs.split(',')).astype(float)
    args.priors = np.array(args.priors.split(',')).astype(float)
    prior = args.priors / np.sum(args.priors)
    contexts = np.array([[1, 1, 0], [1, 0, 1]])
    exp = Experiment(args.alpha, prior, args.costs, contexts, args.n_iter, 50, args.model)
    exp.run()
    exp.save()