import torch
from src.recognition.canonical_ctc import fold_parenthesis_logits,canonical_teacher_kl,canonical_ctc_inputs
from src.recognition.label_normalization import normalize_parentheses
from src.recognition.data import decode

MAPPING={'(':1,')':2,'（':3,'）':4,'A':5}


def test_pooling_preserves_mass_without_reordering_ids():
    torch.manual_seed(3);x=torch.randn(4,2,6,requires_grad=True)
    p=x.softmax(-1);q=fold_parenthesis_logits(x,MAPPING).softmax(-1)
    assert torch.allclose(q[...,1],p[...,1]+p[...,3],atol=1e-7)
    assert torch.allclose(q[...,2],p[...,2]+p[...,4],atol=1e-7)
    assert torch.allclose(q[...,0],p[...,0]) and torch.allclose(q[...,5],p[...,5])
    assert torch.equal(q[...,3],torch.zeros_like(q[...,3]))
    assert normalize_parentheses('（A）(())「５」')=='(A)(())「５」'


def test_ctc_pools_before_collapse_but_preserves_blank_separated_repeats():
    x=torch.full((4,1,6),-20.)
    for t,i in enumerate([1,3,0,3]):x[t,0,i]=20.
    assert decode(x,MAPPING)==['(（（']
    assert decode(fold_parenthesis_logits(x,MAPPING),MAPPING)==['((']


def test_canonical_ctc_and_kl_have_finite_gradients():
    torch.manual_seed(9);x=torch.randn(8,2,6,requires_grad=True);teacher=torch.randn_like(x)
    lengths=torch.tensor([8,6]);targets=torch.tensor([1,2,1,1]);target_lengths=torch.tensor([2,2])
    logp,local=canonical_ctc_inputs(x,targets,MAPPING)
    loss=torch.nn.CTCLoss(blank=0)(logp,local,lengths,target_lengths)
    kd=canonical_teacher_kl(x,teacher,lengths,MAPPING,2.)
    (loss+kd).backward()
    assert torch.isfinite(loss) and torch.isfinite(kd) and torch.isfinite(x.grad).all()
    same=canonical_teacher_kl(x.detach(),x.detach(),lengths,MAPPING,2.)
    assert same.abs()<1e-6


def test_ctc_compaction_does_not_change_saved_or_input_ids():
    targets=torch.tensor([5,1]);before=targets.clone()
    logp,local=canonical_ctc_inputs(torch.randn(4,1,6),targets,MAPPING)
    assert torch.equal(targets,before) and local.tolist()==[3,1]
    assert MAPPING['A']==5 and MAPPING['（']==3 and logp.shape[-1]==4


def test_kl_excludes_artificial_right_padding():
    torch.manual_seed(12);x=torch.randn(8,2,6);teacher=x.clone();lengths=torch.tensor([8,4])
    teacher[4:,1,:]+=torch.randn(4,6)*30
    assert canonical_teacher_kl(x,teacher,lengths,MAPPING).abs()<1e-6
