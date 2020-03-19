from collections import Counter
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: stat_grapheme.py <trans> <vocab>""")
    parser.add_argument("text", help="path to text.")
    parser.add_argument("vocab", help="path to store vocab.")
    parser.add_argument("--vocab-size", type=int, default=100000, help="vocabulary size.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    fn = args.text
    fnw = args.vocab
    vocabsize = args.vocab_size
    txt = ""
    with open(fn, 'r', encoding="utf8") as f:
        for line in f:
            items = line.strip().split(' ', 1)
            if len(items) == 1:
                continue 
            txt += items[1]

    txtlist = list(txt)

    cnter = Counter(txtlist)

    most = cnter.most_common(None)

    with open(fnw, 'w', encoding="utf8") as f:
        t = [m[0] for m in most]
        f.write("\n".join(t[:vocabsize]))



