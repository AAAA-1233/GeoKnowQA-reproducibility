#!/usr/bin/env python3
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    with open(args.before, encoding='utf-8') as f:
        before = json.load(f)
    with open(args.after, encoding='utf-8') as f:
        after = json.load(f)
    names = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge-1', 'rouge-2', 'rouge-l', 'distinct_1', 'distinct_2', 'token_f1', 'exact_match']
    report = {'before': before, 'after': after, 'delta': {k: after[k] - before[k] for k in names}}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report['delta'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
