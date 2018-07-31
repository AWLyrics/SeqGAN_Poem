import numpy as np

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
        else:
            words.append("<UNK>")
    return "".join(words)


with open("./result/test_ret.txt", "r") as f:
    for l in f:
        l = l.split()
        line= [int(x) for x in l]
        print(translate(line))
        # print(line)
# print(translate([58,59, 3,60, 17 ,5 ,8 ,5 ,1 ,61, 62, 3, 63, 7 ,9 ,64 ,18, 3 ,19, 3]))
# with open(translate_file, "r") as f:
#     for l in f:
#         print(translate(l))
