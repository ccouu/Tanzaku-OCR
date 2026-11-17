import itertools
import math
import numpy as np
from src.recognition.prefix_beam import decode_beams,MenuLexiconScore


def collapse(path):
    return ''.join('AB'[c-1] for i,c in enumerate(path) if c and (not i or c!=path[i-1]))


def test_aggregates_all_paths_including_repeats():
    p=np.array([[.25,.6,.15],[.5,.4,.1],[.2,.5,.3],[.4,.1,.5]])
    expected={}
    for path in itertools.product(range(3),repeat=4):
        text=collapse(path);expected[text]=expected.get(text,0.)+np.prod([p[t,c] for t,c in enumerate(path)])
    actual=dict(decode_beams(np.log(p[:,0]),np.tile([1,2],(4,1)),np.log(p[:,1:]),100,{'A':1,'B':2}))
    for text,prob in expected.items():assert abs(math.exp(actual[text])-prob)<1e-12


def test_blank_separates_repeated_characters():
    p=np.array([[.01,.99],[.99,.01],[.01,.99]])
    assert decode_beams(np.log(p[:,0]),np.ones((3,1),dtype=np.int64),np.log(p[:,1:]),5,{'A':1})[0][0]=='AA'


def test_lexicon_allows_unknown_and_locks_digits():
    lex=MenuLexiconScore(['焼鳥','焼鳥定食'])
    assert lex.rerank([('未知語',-1.),('焼鳥',-10.)],.3)=='未知語'
    assert lex.rerank([('焼島680',-1.),('焼鳥880',-.9),('焼鳥680',-1.1)],.3)=='焼鳥680'
    assert lex.coverage('焼鳥定食')==4
