"""Pruned CTC prefix search; prefix probabilities aggregate blank/repeat paths."""
import math
import re
from functools import lru_cache
import numpy as np
from numba import njit


@njit(cache=True)
def logadd(a,b):
    if a == -np.inf:return b
    if b == -np.inf:return a
    return max(a,b)+np.log1p(np.exp(-abs(a-b)))


@njit(cache=True)
def search(blank_logp,token_ids,token_logp,beam_width,relative_cutoff=12.):
    # Encode token IDs as Unicode codepoints, avoiding ambiguity in repeats.
    beams={'':(0.,-np.inf)}
    for t in range(len(blank_logp)):
        nxt={'':(-np.inf,-np.inf)}
        frame_best=max(blank_logp[t],token_logp[t,0])
        for prefix,state in beams.items():
            pb,pnb=state;total=logadd(pb,pnb)
            old=nxt[prefix] if prefix in nxt else (-np.inf,-np.inf)
            nxt[prefix]=(logadd(old[0],total+blank_logp[t]),old[1])
            for j in range(token_ids.shape[1]):
                lp=token_logp[t,j]
                if lp < frame_best-relative_cutoff:continue
                c=chr(token_ids[t,j]);repeated=len(prefix)>0 and prefix[-1]==c
                if repeated:
                    old=nxt[prefix] if prefix in nxt else (-np.inf,-np.inf)
                    nxt[prefix]=(old[0],logadd(old[1],pnb+lp))
                target=prefix+c;score=(pb if repeated else total)+lp
                old=nxt[target] if target in nxt else (-np.inf,-np.inf)
                nxt[target]=(old[0],logadd(old[1],score))
        keys=list(nxt.keys());scores=np.empty(len(keys))
        for i in range(len(keys)):scores[i]=logadd(nxt[keys[i]][0],nxt[keys[i]][1])
        order=np.argsort(scores)
        beams={keys[order[-1]]:nxt[keys[order[-1]]]}
        for i in range(2,min(beam_width,len(keys))+1):
            key=keys[order[-i]];beams[key]=nxt[key]
    keys=list(beams.keys());scores=np.empty(len(keys))
    for i in range(len(keys)):scores[i]=logadd(beams[keys[i]][0],beams[keys[i]][1])
    order=np.argsort(scores)[::-1]
    return [keys[i] for i in order],[scores[i] for i in order]


def decode_beams(blank,ids,lp,width,mapping):
    prefixes,scores=search(np.asarray(blank,dtype=np.float64),np.asarray(ids,dtype=np.int64),np.asarray(lp,dtype=np.float64),width)
    reverse={v:k for k,v in mapping.items()}
    return [(''.join(reverse[ord(c)] for c in prefix),float(score)) for prefix,score in zip(prefixes,scores)]


class MenuLexiconScore:
    """Bounded covered-character bonus; unknown text is allowed with zero bonus."""
    def __init__(self,terms):
        self.terms=frozenset(t for t in terms if 2<=len(t)<=24 and all(c.isalpha() or c in 'ー々ヶ' for c in t))

    @lru_cache(maxsize=200000)
    def coverage(self,text):
        dp=[0]*(len(text)+1)
        for i in range(len(text)):
            dp[i+1]=max(dp[i+1],dp[i])
            for j in range(i+2,min(len(text),i+24)+1):
                if text[i:j] in self.terms:dp[j]=max(dp[j],dp[i]+j-i)
        return min(10,dp[-1])

    def rerank(self,nbest,weight):
        # Dictionary preference cannot change a price/digit sequence relative
        # to acoustic beam25. Beam search itself remains unrestricted.
        digits=''.join(c for c in nbest[0][0] if c.isdigit())
        eligible=[x for x in nbest if ''.join(c for c in x[0] if c.isdigit())==digits]
        return max(eligible,key=lambda x:x[1]+weight*self.coverage(x[0]))[0]
