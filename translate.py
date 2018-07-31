import numpy as np
import argparse

word2idx = np.load("./data/w2i.npy")
word2idx = word2idx.item()
id2word = {k: v for v, k in zip(word2idx.keys(), word2idx.values())}


def translate(word_indexs):
    words = []
    for idx in word_indexs:
        idx = int(idx)
        word = id2word.get(idx)
        #         print(word)
        if word:
            words.append(id2word.get(idx))
        # else:
        #     words.append("<UNK>")
    return "".join(words)


def translate_file(file_name):
    translated = []
    with open(file_name, "r") as f:
        for l in f:
            l = l.split()
            line = [int(x) for x in l]
            # trans = translate(line)
            translated.append(translate(line))
        # print(line)
    return translated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', type=str, help='index file')
    parser.add_argument('--refer_file', default=None, type=str, help='reference file')
    parser.add_argument('--input_file', default=None, type=str, help='input file')

    args = parser.parse_args()

    if args.refer_file and args.input_file:
        trans = translate_file(args.index_file)
        refer = translate_file(args.refer_file)
        input = translate_file(args.input_file)
        for t, r, i in zip(trans, refer, input):
            print("input: ", i)
            print("translate: ", t)
            print("refer: ", r)
            print()
    else:
        for i in translate_file(args.index_file):
            print(i)

# print(translate([58,59, 3,60, 17 ,5 ,8 ,5 ,1 ,61, 62, 3, 63, 7 ,9 ,64 ,18, 3 ,19, 3]))
# with open(translate_file, "r") as f:
#     for l in f:
#         print(translate(l))
