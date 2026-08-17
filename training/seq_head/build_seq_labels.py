"""
train_CoT.json -> per-organ 시퀀스 어휘 + case별 GT seq_id.

시퀀스 = 한 케이스의 CoT 질문 순서(문자열 리스트). organ별로 유니크한 시퀀스를
빈도순으로 모아 seq_id 부여. (Pathwise prepare_report_to_path.py 와 동일한 발상.)

출력:
  artifacts/seq_vocab.json   {organ: [[q1,q2,...], ...]}   # 빈도 내림차순
  artifacts/seq_labels.json  {stem: {"organ":organ, "seq_id":int}}
"""
import json
from collections import Counter, defaultdict

from nar2_common import TRAIN_COT, ART, chain_questions, ORGANS

data = json.load(open(TRAIN_COT))
print(f"[labels] {len(data)} cases")

# organ -> Counter(tuple(questions))
seqs_by_organ = defaultdict(Counter)
case_seq = {}  # stem -> (organ, tuple(questions))
skipped = 0
for c in data:
    organ = c.get("organ", "").strip().lower()
    if organ not in ORGANS:
        skipped += 1
        continue
    qs = tuple(chain_questions(c["chain-of-thought"]))
    if not qs:
        skipped += 1
        continue
    stem = c["id"].replace(".tiff", "")
    seqs_by_organ[organ][qs] += 1
    case_seq[stem] = (organ, qs)

# organ별 어휘(빈도순) + tuple->id
vocab = {}
seq_to_id = {}
for organ in ORGANS:
    ordered = [s for s, _ in seqs_by_organ[organ].most_common()]
    vocab[organ] = [list(s) for s in ordered]
    seq_to_id[organ] = {s: i for i, s in enumerate(ordered)}
    print(f"  {organ:9s}: {sum(seqs_by_organ[organ].values()):5d} cases, "
          f"{len(ordered):3d} unique sequences")

labels = {}
for stem, (organ, qs) in case_seq.items():
    labels[stem] = {"organ": organ, "seq_id": seq_to_id[organ][qs]}

json.dump(vocab, open(f"{ART}/seq_vocab.json", "w"), ensure_ascii=False, indent=1)
json.dump(labels, open(f"{ART}/seq_labels.json", "w"), ensure_ascii=False)
print(f"[labels] skipped {skipped}; wrote seq_vocab.json, seq_labels.json "
      f"({len(labels)} labeled cases)")
