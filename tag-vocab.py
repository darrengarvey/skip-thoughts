#!/usr/bin/env python

"""
Take a vocab file (one word per line) and output an annotated vocab file
that includes the word category from WordNet.

The format is compatible with tensorboard's embeddings tab.
"""

import argparse
import sys
from nltk.corpus import wordnet as wn


def parse_args(args):
    parser = argparse.ArgumentParser('Tag a vocab file with the word categories.')
    parser.add_argument('-i', '--input', required=True, help='Input file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    counts = [0, 0]
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        fout.write('Word\tCategory\n')
        for line in fin:
            word = line.strip()
            if len(wn.synsets(word)):
                counts[0] += 1
                fout.write('{}\t{}\n'.format(word, wn.synsets(word)[0].lexname()))
            else:
                counts[1] += 1
                fout.write('{}\n'.format(word))
    print ('Read {} words ({} untagged, {} tagged)'.format(
        counts[0] + counts[1], counts[0], counts[1]))

if __name__ == '__main__':
    main()
