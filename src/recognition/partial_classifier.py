"""Train an explicit subset of CTC rows without optimizer drift in frozen rows."""
import torch
from torch import nn


class SelectedCTCRows(nn.Module):
    def __init__(self,base,ids):
        super().__init__()
        if not ids or len(set(ids))!=len(ids) or min(ids)<1 or max(ids)>=base.out_channels:
            raise ValueError('Selected rows must be unique nonblank classifier IDs')
        self.base=base.requires_grad_(False)
        self.register_buffer('indices',torch.tensor(ids,dtype=torch.long))
        self.rows=nn.Conv1d(base.in_channels,len(ids),1)
        with torch.no_grad():
            self.rows.weight.copy_(base.weight[ids]);self.rows.bias.copy_(base.bias[ids])

    def forward(self,features):
        return self.base(features).index_copy(1,self.indices,self.rows(features))


class GroupedCTCRows(nn.Module):
    """Disjoint row groups with independent optimizer learning rates."""
    def __init__(self,base,groups):
        super().__init__();self.base=base.requires_grad_(False);self.heads=nn.ModuleDict()
        all_ids=[i for ids in groups.values() for i in ids]
        if not all_ids or len(set(all_ids))!=len(all_ids) or min(all_ids)<1 or max(all_ids)>=base.out_channels:
            raise ValueError('Groups must contain disjoint nonblank IDs')
        for name,ids in groups.items():
            self.register_buffer('indices_'+name,torch.tensor(ids,dtype=torch.long))
            head=nn.Conv1d(base.in_channels,len(ids),1)
            with torch.no_grad():head.weight.copy_(base.weight[ids]);head.bias.copy_(base.bias[ids])
            self.heads[name]=head

    def forward(self,features):
        logits=self.base(features)
        for name,head in self.heads.items():logits=logits.index_copy(1,getattr(self,'indices_'+name),head(features))
        return logits


def export_state(model):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if isinstance(model.classifier,GroupedCTCRows):
        for suffix in ('weight','bias'):
            original=state.pop('classifier.base.'+suffix)
            for name in model.classifier.heads:
                original.index_copy_(0,state['classifier.indices_'+name],state.pop('classifier.heads.'+name+'.'+suffix))
            state['classifier.'+suffix]=original
        for name in model.classifier.heads:state.pop('classifier.indices_'+name)
    if isinstance(model.classifier,SelectedCTCRows):
        ids=state.pop('classifier.indices')
        for name in ('weight','bias'):
            original=state.pop('classifier.base.'+name)
            original.index_copy_(0,ids,state.pop('classifier.rows.'+name))
            state['classifier.'+name]=original
    return state
