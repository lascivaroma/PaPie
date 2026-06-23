"""
Compare inference variants (fp32 baseline / INT8 quantized / char-embedding
cache) of a PaPie model on a gold TSV file: accuracy vs gold, agreement between
variants, and wall-clock speed.

The TSV must have a header row; each requested task is scored against the gold
column whose name matches the task name (e.g. task "pos" -> column "pos").

Usage
=====
    python scripts/compare_inference.py "Thomas More --  Utopia.tsv" \
        --model lasla-plus-lemma.tar:lemma \
        --model lasla-plus-pos.tar:pos \
        --limit 4000 --sent-len 20

Notes
=====
- Sentences are formed by fixed-size chunking of the token stream (--sent-len).
  This is applied identically to every variant, so the reported *delta* between
  variants is valid even though the absolute accuracy depends on the chunking
  and on any domain mismatch between the model and the test text.
- The cache is expected to be loss-free: its predictions are asserted identical
  to the fp32 baseline.
"""

import os
import sys
import time
import argparse

# allow running as `python scripts/compare_inference.py` without installing pie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pie.tagger import Tagger


def read_tsv(path, limit=None):
    """Return (header, list-of-rows) where each row is a list of column values."""
    with open(path) as f:
        header = next(f).rstrip("\n").split("\t")
        rows = []
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) == len(header):
                rows.append(cells)
            if limit and len(rows) >= limit:
                break
    return header, rows


def chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def predict(model_path, task, forms, sents, lengths,
            quantize=False, cache=False, device="cpu"):
    """Tag `sents` and return (flat predictions, elapsed seconds)."""
    tagger = Tagger(device=device, quantize=quantize, cache=cache)
    tagger.add_model(model_path, task)
    start = time.perf_counter()
    tagged, _ = tagger.tag(sents, lengths)
    elapsed = time.perf_counter() - start
    preds = [tags[0] for sent in tagged for (_tok, tags) in sent]
    return preds, elapsed


def accuracy(pred, gold):
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tsv", help="gold TSV file (with header)")
    parser.add_argument("--model", action="append", required=True,
                        metavar="PATH:TASK",
                        help="model tarball and task, e.g. lasla-plus-pos.tar:pos "
                             "(repeatable)")
    parser.add_argument("--limit", type=int, default=4000,
                        help="max tokens to evaluate (the lemma decoder is slow on CPU)")
    parser.add_argument("--sent-len", type=int, default=20,
                        help="fixed chunk size used to form pseudo-sentences")
    parser.add_argument("--device", default="cpu",
                        help="torch device, e.g. cpu or cuda (default: cpu). "
                             "Note: INT8 dynamic quantization is CPU-only.")
    args = parser.parse_args()

    header, rows = read_tsv(args.tsv, args.limit)
    forms = [r[0] for r in rows]
    sents = chunk(forms, args.sent_len)
    lengths = [len(s) for s in sents]
    print(f"file: {args.tsv}")
    print(f"tokens: {len(forms)}  sentences: {len(sents)}  (chunk={args.sent_len})\n")

    for spec in args.model:
        model_path, _, task = spec.partition(":")
        if not task:
            parser.error(f"--model must be PATH:TASK, got {spec!r}")
        if task not in header:
            parser.error(f"task {task!r} is not a column in the TSV header {header}")
        gold = [r[header.index(task)] for r in rows]

        fp32, t_fp32 = predict(model_path, task, forms, sents, lengths,
                               device=args.device)
        int8, t_int8 = predict(model_path, task, forms, sents, lengths,
                               quantize=True, device=args.device)
        cached, t_cache = predict(model_path, task, forms, sents, lengths,
                                  cache=True, device=args.device)
        both, t_both = predict(model_path, task, forms, sents, lengths,
                               quantize=True, cache=True, device=args.device)

        ntok = len(fp32)

        def spd(t):
            return f"{ntok / t:6.0f} tok/s"

        def up(t):
            return f"{t_fp32 / t:.2f}x ({(t_fp32 / t - 1) * 100:+.0f}%)"

        print(f"===== {task}  ({model_path}) =====")
        print(f"  accuracy   fp32={accuracy(fp32, gold):.4f}   "
              f"int8={accuracy(int8, gold):.4f} ({accuracy(int8, gold) - accuracy(fp32, gold):+.4f})   "
              f"cache={accuracy(cached, gold):.4f} ({accuracy(cached, gold) - accuracy(fp32, gold):+.4f})   "
              f"int8+cache={accuracy(both, gold):.4f} ({accuracy(both, gold) - accuracy(fp32, gold):+.4f})")
        print(f"  agreement  fp32-vs-int8={agreement(fp32, int8):.4f}   "
              f"fp32-vs-cache={agreement(fp32, cached):.4f}   "
              f"fp32-vs-int8+cache={agreement(fp32, both):.4f}")
        print(f"  speed      fp32={spd(t_fp32)}   int8={spd(t_int8)}   "
              f"cache={spd(t_cache)}   int8+cache={spd(t_both)}")
        print(f"  speedup    fp32=1.00x   int8={up(t_int8)}   "
              f"cache={up(t_cache)}   int8+cache={up(t_both)}")
        for name, preds in (("cache", cached), ("int8+cache", both)):
            # cache must be loss-free relative to its non-cached counterpart
            ref = fp32 if name == "cache" else int8
            if preds == ref:
                print(f"  {name} is loss-free vs {'fp32' if name == 'cache' else 'int8'} "
                      f"(identical predictions) ✓")
            else:
                print(f"  WARNING: {name} differs from "
                      f"{'fp32' if name == 'cache' else 'int8'} on "
                      f"{sum(a != b for a, b in zip(ref, preds))} tokens")
        print()


if __name__ == "__main__":
    main()
