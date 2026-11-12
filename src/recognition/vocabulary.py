"""Append-only CTC vocabulary extension and independently trained new rows."""
import copy
import torch
from torch import nn


def extend_vocabulary(source,characters):
    vocab=copy.deepcopy(source);mapping=dict(vocab['char_to_id'])
    assert vocab.get('blank_id',0)==0 and set(mapping.values())==set(range(1,len(mapping)+1))
    added=sorted(set(characters)-set(mapping),key=ord)
    for c in added:mapping[c]=len(mapping)+1
    vocab.update(char_to_id=mapping,characters=[c for c,_ in sorted(mapping.items(),key=lambda x:x[1])],
                 base_minimum_frequency=vocab.get('minimum_frequency'),minimum_frequency=None,
                 extension=dict(policy='Append only; old IDs and blank0 preserved',added_characters=added))
    return vocab,added


class ExtendedCTCHead(nn.Module):
    """Separate new classifier rows permit a higher LR without changing old rows."""
    def __init__(self,old,extra,*,initialization='constant',blank_margin=8.):
        super().__init__();self.old=old
        self.new=nn.Conv1d(old.in_channels,extra,1)
        if initialization=='constant':
            # Historical V40 behavior, retained only for reproducibility.
            nn.init.zeros_(self.new.weight);nn.init.constant_(self.new.bias,-12.)
        elif initialization=='below_blank':
            if blank_margin<=0:raise ValueError('New logits must start below blank')
            with torch.no_grad():
                self.new.weight.copy_(old.weight[:1].expand(extra,-1,-1))
                self.new.bias.copy_(old.bias[:1].expand(extra)-blank_margin)
        else:raise ValueError('Unknown new-row initialization')

    def forward(self,features):return torch.cat((self.old(features),self.new(features)),dim=1)


def fused_state(model):
    """Export standard MenuRecognizer keys: no custom classifier needed to load."""
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for suffix in ('weight','bias'):
        old=state.pop('classifier.old.'+suffix);new=state.pop('classifier.new.'+suffix)
        state['classifier.'+suffix]=torch.cat((old,new))
    return state


def split_existing_classifier(full,old_classes):
    """Split already-trained rows for separate LRs without reinitializing any."""
    if full.kernel_size!=(1,) or full.bias is None or not 0<old_classes<full.out_channels:
        raise ValueError('Expected a biased1x1CTCclassifier and valid split')
    old=nn.Conv1d(full.in_channels,old_classes,1)
    head=ExtendedCTCHead(old,full.out_channels-old_classes).to(full.weight)
    with torch.no_grad():
        head.old.weight.copy_(full.weight[:old_classes]);head.old.bias.copy_(full.bias[:old_classes])
        head.new.weight.copy_(full.weight[old_classes:]);head.new.bias.copy_(full.bias[old_classes:])
    return head
