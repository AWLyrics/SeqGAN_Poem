import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np

def load_data(file_name):
    i = 0
    train = []
    with open(file_name, "r") as f:
        for l in f:
            l = l.strip()
            train.append(list(jieba.cut(l)))

    tokenizer = Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    # print(tokenizer.word_index)
    # print(list(map(lambda x: len(x), train_idx)))
    train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_idx, maxlen=20, padding='post',
                                                                 truncating='post')
    # print(train_padded)
    return train_padded, tokenizer.word_index


train_idx, word2idx = load_data("rb_lyric.txt")
# print(train_idx)
print(len(word2idx))
np.save("lyric_w2i.npy", word2idx)



with open("lyric.txt", "w") as f:
    for l in train_idx:
        line = [str(i)+' ' for i in l]
        f.write("".join(line) )
        f.write('\n')
