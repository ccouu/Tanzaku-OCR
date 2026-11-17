"""Canonical parenthesis equivalence without changing saved vocabulary IDs."""
from collections import Counter,defaultdict
import torch
from torch.nn import functional as F
from .label_normalization import normalize_parentheses
from .data import decode,collate,minimum_ctc_steps
from .metrics import align_errors,summarise


def fold_parenthesis_logits(logits,mapping,drop_aliases=False):
    """Pool each width pair's probability mass before CTC collapse/normalization.

    logaddexp on raw logits preserves the full softmax denominator. Aliases have
    no remaining mass; physical classifier rows/IDs and blank0 stay unchanged.
    """
    x=logits.float();aliases=[];columns=list(x.unbind(-1))
    for wide,ascii_char in [('（','('),('）',')')]:
        a,b=mapping[wide],mapping[ascii_char]
        columns[b]=torch.logaddexp(x[...,a],x[...,b])
        columns[a]=torch.full_like(x[...,a],float('-inf'));aliases.append(a)
    result=torch.stack(columns,-1)
    if drop_aliases:
        keep=[i for i in range(x.shape[-1]) if i not in aliases]
        result=result[...,keep]
    return result


def canonical_teacher_kl(logits,teacher,lengths,mapping,temperature=2.):
    """Full-softmax KL on semantic classes, including all new Latin mass.

    Pool at the actual distillation temperature; omit zero-mass alias columns
    to avoid0*infinity. This never renormalizes over only old classifier rows.
    """
    student=fold_parenthesis_logits(logits.float()/temperature,mapping,True).log_softmax(-1)
    target=fold_parenthesis_logits(teacher.detach().float()/temperature,mapping,True).softmax(-1)
    losses=F.kl_div(student,target,reduction='none').sum(-1)*temperature**2
    valid=torch.arange(logits.shape[0],device=logits.device)[:,None]<lengths.to(logits.device)[None,:]
    return losses[valid].mean()


def canonical_ctc_inputs(logits,targets,mapping):
    """Compact only the loss tensor, keeping vocabulary/checkpoint IDs intact.

    PyTorch CTC backward can yieldNaNs for explicit-negative-infinity alias
    classes. Remove those two zero-mass columns and remap targets locally.
    """
    aliases={mapping['（'],mapping['）']}
    keep=[i for i in range(logits.shape[-1]) if i not in aliases]
    lookup=targets.new_full((logits.shape[-1],),-1)
    lookup[torch.tensor(keep,device=targets.device)]=torch.arange(len(keep),device=targets.device)
    local=lookup[targets]
    if (local<0).any():raise ValueError('CTC targets must canonicalize full-width parentheses first')
    return fold_parenthesis_logits(logits,mapping,True).log_softmax(-1),local


class CanonicalTrainingLabels:
    """Return canonical targets without modifying images or source label rows."""
    def __init__(self,mix,mapping):self.mix=mix;self.mapping=mapping
    def __getitem__(self,key):
        image,targets,orientation,width,text,compressed=self.mix[key]
        text=normalize_parentheses(text)
        if (width+3)//4<minimum_ctc_steps(text):raise ValueError('Canonical CTC alignment infeasible')
        targets=torch.tensor([self.mapping[c] for c in text],dtype=torch.long)
        return image,targets,orientation,width,text,compressed


def summarize_predictions(predictions):
    fonts=sorted({r['font'] for r in predictions if r['font']})
    by_font={f:summarise([r for r in predictions if r['font']==f]) for f in fonts}
    cohorts={c:[v['cer'] for f,v in by_font.items() if any(r['font']==f and r['cohort']==c for r in predictions)] for c in ('old','new')}
    return {'0.0':dict(all=summarise(predictions),predictions=predictions,
        by_length={str(n):summarise([r for r in predictions if len(r['truth'])==n]) for n in sorted({len(r['truth']) for r in predictions})},
        by_font=by_font,cohort_macro={c:sum(v)/len(v) for c,v in cohorts.items() if v},
        macro_font_cer=sum(v['cer'] for v in by_font.values())/len(by_font) if by_font else None,
        by_cohort={c:summarise([r for r in predictions if r['cohort']==c]) for c in sorted({r['cohort'] for r in predictions if r['cohort']})})}


def evaluate_parenthesis_modes(model,ds,device,precision):
    """One forward pass yields literal, text-normalized and pooled-CTC scores."""
    groups=defaultdict(list);records={k:{} for k in ('literal','text_normalized','canonical')};model.eval()
    for i in range(len(ds)):groups[ds.canvas_at(i)].append(i)
    def record(i,truth,pred):
        errors=Counter(o['type'] for o in align_errors(truth,pred) if o['type']!='match');row=ds.rows[i]
        return dict(index=i,truth=truth,prediction=pred,font=row.get('preferred_font'),cohort=row.get('cohort'),
            error_counts={k:errors[k] for k in ('deletion','insertion','substitution')})
    with torch.inference_mode():
        for canvas,indices in sorted(groups.items()):
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];batch=collate([ds[i] for i in ids],padded_width=canvas)
                with torch.autocast('cuda',dtype=torch.float16 if precision=='fp16' else torch.bfloat16):
                    logits=model(batch['images'].to(device),batch['orientation'].to(device),batch['input_lengths'])
                raw=decode(logits,model.vocabulary,batch['input_lengths'])
                canonical=decode(fold_parenthesis_logits(logits,model.vocabulary),model.vocabulary,batch['input_lengths'])
                for i,truth,pred,pooled in zip(ids,batch['texts'],raw,canonical):
                    records['literal'][i]=record(i,truth,pred)
                    records['text_normalized'][i]=record(i,normalize_parentheses(truth),normalize_parentheses(pred))
                    records['canonical'][i]=record(i,normalize_parentheses(truth),pooled)
    return {mode:summarize_predictions([rr[i] for i in range(len(ds))]) for mode,rr in records.items()}
