"""Versioned acceptance adjustment over the preserved legacy gate results."""
import copy
import json
from pathlib import Path

POLICY_PATH=Path(__file__).with_name('retention_policy.json')


def apply_retention_policy(legacy,policy=None):
    policy=policy or json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    result=copy.deepcopy(legacy)
    if result.get('policy_id')==policy['id']:
        return result
    if 'policy_id' in result:
        raise ValueError('Apply the new policy to original gate results, not another adjusted policy')
    result['original_caps']=copy.deepcopy(result['caps'])
    result['original_failed']=list(result['failed'])
    for name,allowance in policy['additional_cap_absolute'].items():
        if allowance<0 or name not in result['caps'] or name not in result['current']:
            raise ValueError('Invalid retention policy cap adjustment')
        result['caps'][name]+=allowance
    result['failed']=[name for name,value in result['current'].items() if value>result['caps'][name]+1e-12]
    result['eligible']=not result['failed'];result['policy_id']=policy['id']
    return result
