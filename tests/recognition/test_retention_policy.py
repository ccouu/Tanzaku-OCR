import pytest
from src.recognition.retention import apply_retention_policy


def original(two=.14,one=.30):
    return dict(current={'pooled_original_2char':two,'pooled_original_1char':one,'other':.05},
                caps={'pooled_original_2char':.135,'pooled_original_1char':.30,'other':.06},
                failed=['pooled_original_2char'],eligible=False)


def test_only_two_character_cap_changes_and_input_is_preserved():
    source=original();adjusted=apply_retention_policy(source)
    assert adjusted['eligible'] and adjusted['failed']==[]
    assert adjusted['caps']['pooled_original_2char']==pytest.approx(.14)
    assert adjusted['caps']['pooled_original_1char']==.30 and adjusted['caps']['other']==.06
    assert source['caps']['pooled_original_2char']==.135 and not source['eligible']
    assert apply_retention_policy(adjusted)==adjusted


def test_one_character_regression_still_fails():
    adjusted=apply_retention_policy(original(one=.305))
    assert adjusted['failed']==['pooled_original_1char'] and not adjusted['eligible']


def test_two_character_errors_above_new_cap_still_fail():
    assert not apply_retention_policy(original(two=.1425))['eligible']
