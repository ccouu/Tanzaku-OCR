"""Preserve both reviews; approve conservative agreement plus sourced essentials."""
from collections import Counter
import json
from pathlib import Path
import hashlib
import shutil

P=Path(__file__).parent


def read(name):return [json.loads(s) for s in (P/name).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(name,rows):(P/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def main():
    candidates=read('candidates.jsonl');parent={r['id']:r for r in read('parent_review.jsonl')};argon={r['id']:r for r in read('argon_review.jsonl')}
    assert len(candidates)==len(parent)==len(argon)==5069
    assert {r['id'] for r in candidates}==parent.keys()==argon.keys()
    approved=[];reviewed=[];missing=[];disagreement=[]
    for row in candidates:
        a,b=parent[row['id']],argon[row['id']]
        assert a['term']==row['normalized'] and b['term']==row['normalized']
        agree=a['classification']==b['classification']=='practical_menu' and a['confidence']=='high' and b['confidence']=='high'
        item=dict(row,parent_review=a,argon_review=b,decision='approved' if agree else 'review_queue',review_method='independent LLM agreement')
        reviewed.append(item)
        if agree:
            (approved if row['in_vocab'] else missing).append(item)
        elif a['classification']=='practical_menu' or b['classification']=='practical_menu':disagreement.append(item)
        if not row['in_vocab'] and not agree:missing.append(item)
    sources=dict(currency='https://kotobank.jp/word/%E5%86%86-37934',
        basic='https://gunma-insyoku.or.jp/documents/itadakimasu.pdf',
        portions='https://www.yoshinoya.com/menu/gyudon/gyu-don/',
        prices='https://www.yoshinoya.com/wp-content/uploads/2024/12/02085412/news_2024120202.pdf',
        service='https://e-shop.matsuyafoods.co.jp/collections/single',
        tax_excluded='https://kurakuracafe.com/?page_id=13%2F')
    # These are separately verified surface forms, not undocumented database edits.
    supplements=[('円','えん','yen','currency'),('水','みず','water','basic'),('氷','こおり','ice','basic'),('貝','かい','shellfish','basic'),
        ('税込','ぜいこみ','tax included','prices'),('税別','ぜいべつ','tax excluded','tax_excluded'),('大盛','おおもり','large serving','portions'),('並盛','なみもり','regular serving','portions'),
        ('特盛','とくもり','extra-large serving','portions'),('無料','むりょう','free of charge','service'),('単品','たんぴん','single item','service')]
    vocab=json.loads(Path('data/datasets/recognition/v27_fonts/vocab.json').read_text(encoding='utf-8'))['char_to_id']
    existing={r['normalized'] for r in approved}
    for i,(term,reading,meaning,source) in enumerate(supplements):
        if term in existing:continue
        row=dict(id=f'supplement:{i}',canonical_jp=term,normalized=term,reading_hiragana=reading,name_en=meaning,
            length=len(term),source=sources[source],in_vocab=all(c in vocab for c in term),
            missing_characters=sorted(set(term)-set(vocab)),decision='approved',
            review_method='parent LLM essential term; independent source check; not claimed dual-model agreement')
        (approved if row['in_vocab'] else missing).append(row);existing.add(term)
    # Deduplicate normalized forms without silently rewriting historical source entries.
    canonical={}
    for row in approved:canonical.setdefault(row['normalized'],row)
    approved=sorted(canonical.values(),key=lambda r:(r['length'],r['normalized']))
    assert all(1<=len(r['normalized'])<=2 and all(c in vocab for c in r['normalized']) for r in approved)
    assert all(any(r['normalized']==c for r in approved) for c in ('円','水','氷','貝'))
    write('reviewed_candidates.jsonl',reviewed);write('approved_terms.jsonl',approved)
    write('disagreements.jsonl',disagreement);write('vocabulary_gaps.jsonl',missing)
    (P/'approved_1char.txt').write_text('\n'.join(r['normalized'] for r in approved if r['length']==1)+'\n',encoding='utf-8')
    (P/'approved_2char.txt').write_text('\n'.join(r['normalized'] for r in approved if r['length']==2)+'\n',encoding='utf-8')
    counts=dict(candidates=len(candidates),approved=len(approved),by_length=dict(Counter(r['length'] for r in approved)),
        disagreements=len(disagreement),vocabulary_gap_entries=len(missing),
        argon_counts=dict(Counter(r['classification'] for r in argon.values())),
        parent_counts=dict(Counter(r['classification'] for r in parent.values())),
        approved_sha256=hashlib.sha256((P/'approved_terms.jsonl').read_bytes()).hexdigest(),
        database_unchanged=True,model_requested='agy --model argon --effort medium',
        dictionary_sha256=hashlib.sha256(Path('data/processed/lexicon/culinary_menu.db').read_bytes()).hexdigest(),
        license='CC-BY-SA-4.0; original JMdict attribution retained in JMdict-LICENSE.html',
        limits='Conservative reviewed subset,not an exhaustive list of all meaningful Japanese terms. Parent leaves non-shortlisted items uncertain. Gemini methodology is documented in argon_summary.md. Not all dictionary glosses are correct surface senses.')
    (P/'manifest.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile('data/processed/lexicon/JMdict-LICENSE.html',P/'JMdict-LICENSE.html')
    lines=['# Reviewed short menu terms','',f"Decision records cover{len(candidates)} exported candidates; conservative approved vocabulary contains{len(approved)} terms. Unresolved entries remain uncertain,not exhaustively validated.",
        '',f"Length counts:{counts['by_length']}. Disagreements:{len(disagreement)}. Vocabulary-gap entries:{len(missing)}.",
        '', 'Training uses only approved_terms.jsonl. Main dictionary remains untouched. Approved dictionary candidates require high-confidence practical-menu agreement between the parent LLM and Gemini Argon. Separately sourced essentials are marked explicitly. Archaic/uncertain/irrelevant entries are retained for audit,not deleted.',
        '', 'This is not a claim that every remaining dictionary entry is invalid. Review queues need further lexical work. Source meanings and readings may reflect homographs; corpus-based menu frequency has not been measured.',
        '', 'approved_1char.txt / approved_2char.txt are the readable lists. reviewed_candidates.jsonl preserves both decisions and original metadata. disagreements.jsonl and vocabulary_gaps.jsonl are separate follow-up lists.']
    (P/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':main()
