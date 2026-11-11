"""Minimum-edit recognition diagnostics and additive blank-penalty decoding."""
from collections import Counter
import math
from .data import decode

def align_errors(truth, prediction):
    """Align truth to prediction: minimum edits, then maximum exact matches.

    Remaining ties prefer diagonal, deletion, insertion. Positions are zero based;
    insertion reference positions identify the gap before that reference character.
    A minimum-edit alignment can be ambiguous; operations are diagnostic labels.
    """
    matrix = [[(j, 0) for j in range(len(prediction) + 1)]]
    for i, ca in enumerate(truth, 1):
        row = [(i, 0)]
        for j, cb in enumerate(prediction, 1):
            left, up, diagonal = row[-1], matrix[-1][j], matrix[-1][j-1]
            row.append(min((left[0]+1, left[1]), (up[0]+1, up[1]),
                           (diagonal[0]+(ca != cb), diagonal[1]-(ca == cb))))
        matrix.append(row)
    operations = []
    i, j = len(truth), len(prediction)
    while i or j:
        if i and j and matrix[i][j] == (
                matrix[i-1][j-1][0]+(truth[i-1] != prediction[j-1]),
                matrix[i-1][j-1][1]-(truth[i-1] == prediction[j-1])):
            operations.append(dict(type='match' if truth[i-1] == prediction[j-1] else 'substitution',
                                   truth=truth[i-1], prediction=prediction[j-1],
                                   reference_position=i-1, prediction_position=j-1))
            i, j = i-1, j-1
        elif i and matrix[i][j] == (matrix[i-1][j][0]+1, matrix[i-1][j][1]):
            operations.append(dict(type='deletion', truth=truth[i-1], prediction='',
                                   reference_position=i-1, prediction_position=j))
            i -= 1
        else:
            operations.append(dict(type='insertion', truth='', prediction=prediction[j-1],
                                   reference_position=i, prediction_position=j-1))
            j -= 1
    return list(reversed(operations))


def summarise(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['error_counts'])
    characters = sum(len(r['truth']) for r in rows)
    errors = sum(counts.values())
    return dict(lines=len(rows), characters=characters, exact=sum(r['truth'] == r['prediction'] for r in rows),
                edits=errors, cer=errors/characters if characters else None,
                error_counts={k: counts[k] for k in ('deletion', 'insertion', 'substitution')},
                lines_with_error={k: sum(r['error_counts'][k] > 0 for r in rows)
                                  for k in ('deletion', 'insertion', 'substitution')})

def decode_with_blank_penalty(logits, mapping, lengths=None, penalty=0.0):
    """Subtract a nonnegative additive blank-logit penalty at inference only."""
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Blank penalty must be finite and nonnegative')
    if penalty:
        logits = logits.float().clone()
        logits[...,0] -= penalty
    return decode(logits,mapping,lengths)
