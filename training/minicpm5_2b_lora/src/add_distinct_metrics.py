#!/usr/bin/env python3
"""Add corpus-level Distinct-1/2 to an existing generation metrics JSON."""
import argparse
import json
from pathlib import Path

import jieba


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--metrics', required=True)
    args = p.parse_args()
    unigrams, bigrams = [], []
    with open(args.predictions, encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['prediction'].strip()
            tokens = [x for x in jieba.lcut(text) if x.strip()]
            unigrams.extend(tokens)
            bigrams.extend(zip(tokens, tokens[1:]))
    with open(args.metrics, encoding='utf-8') as f:
        metrics = json.load(f)
    metrics['distinct_1'] = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    metrics['distinct_2'] = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    with open(args.metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'distinct_1': metrics['distinct_1'], 'distinct_2': metrics['distinct_2']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
