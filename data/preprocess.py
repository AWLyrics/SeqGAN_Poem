from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

total = []
x = []
y = []
with open("x.txt", "r") as f:
    for i in f:
        s = i.strip("\n")
        s = s.lower()
        total.append(i)
        # x.append(i)

with open("y.txt", "r") as f:
    for i in f:
        s = i.strip("\n")
        s = s.lower()
        total.append(i)
        # y.append(i)


def write2file(file_name, array):
    with open(file_name, "w") as f:
        for i in array:
            f.write(i)


# print(total[:20])


# x_array = np.asarray(x)
# y_array = np.asarray(y)
# print(len(x_array))
# perm = np.random.permutation(len(x_array))
# #
# # x_shuffled = x_array[perm]
# # y_shuffled = y_array[perm]
# # np.random.seed(88)
# #
# # test_size = 1000
# # dev_size = 1000
#
# # print(x_shuffled[:])
# # print(y_shuffled[:2])
# test_x = x_shuffled[:1000]
# test_y = y_shuffled[:1000]
#
# # print(test_x[:2])
# # print(test_y[:2])
# #
# dev_x = x_shuffled[1000:2000]
# dev_y = y_shuffled[1000:2000]
# # print(dev_x[:2])
# # print(dev_y[:2])
# #
# train_x = x_shuffled[2000:]
# train_y = y_shuffled[2000:]


def write2file(file_name, array):
    with open(file_name, "w") as f:
        for i in array:
            f.write(i)


#
def preprocess(file_name, save_file, tokenizer, maxlen=20):
    text = []
    with open(file_name, "r") as fin:
        for i in fin:
            text.append(i)
        text_idx = tokenizer.texts_to_sequences(text)
        text_padded = pad_sequences(text_idx, maxlen=maxlen, padding="post", truncating="post")
    with open(save_file, "w") as fout:
        for l in text_padded:
            line = "".join([str(x)+" " for x in l])
            fout.write(line)
            fout.write("\n")

tokenizer = Tokenizer(num_words=20000, oov_token='<UNK>')
tokenizer.fit_on_texts(total)
np.save("w2i.npy", tokenizer.word_index)
preprocess("dev_x.txt", "dev_idx_x.txt", tokenizer)
preprocess("dev_y.txt", "dev_idx_y.txt", tokenizer)
preprocess("test_x.txt", "test_idx_x.txt", tokenizer)
preprocess("test_y.txt", "test_idx_y.txt", tokenizer)
preprocess("train_x.txt", "train_idx_x.txt", tokenizer)
preprocess("train_y.txt", "train_idx_y.txt", tokenizer)

# x_idx = tokenizer.texts_to_sequences(x)
# y_idx = tokenizer.texts_to_sequences(y)
# #
# # print(tokenizer.word_index['<UNK>'])
# # print(tokenizer.word_index['<UNK>'])
# x_padded = pad_sequences(x_idx, maxlen=20, padding='post', truncating='post')
# y_padded = pad_sequences(y_idx, maxlen=20, padding='post', truncating='post')
#
#
# with open("x_lyric.txt", 'w') as f:
#     for l in x_padded:
#         line = "".join([str(x) + " " for x in l])
#         f.write(line)
#         f.write("\n")
#
#
# with open("y_lyric.txt", 'w') as f:
#     for l in y_padded:
#         line = "".join([str(x) + " " for x in l])
#         f.write(line)
#         f.write("\n")

# with open()
# print(len(x_padded)) # 92639

# print(max(map(lambda x: len(x), x_idx)))
# print(min(map(lambda x: len(x), x_idx)))
# print(sum(map(lambda x: len(x), x_idx)) / len(x_idx))
