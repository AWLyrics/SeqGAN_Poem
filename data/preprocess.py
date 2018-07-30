from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

total = []
x = []
y = []
with open("x.txt", "r") as f:
    for i in f:
        s = i.strip("\n")
        s = s.lower()
        total.append(i)
        x.append(i)

with open("y.txt", "r") as f:
    for i in f:
        s = i.strip("\n")
        s = s.lower()
        total.append(i)
        y.append(i)


# print(x[:100])
# test = x[:100]
tokenizer = Tokenizer(num_words=20000, oov_token='<UNK>')
tokenizer.fit_on_texts(total)
x_idx = tokenizer.texts_to_sequences(x)
y_idx = tokenizer.texts_to_sequences(y)
#
# print(tokenizer.word_index['<UNK>'])
# print(tokenizer.word_index['<UNK>'])
x_padded = pad_sequences(x_idx, maxlen=20, padding='post', truncating='post')
y_padded = pad_sequences(y_idx, maxlen=20, padding='post', truncating='post')


with open("x_lyric.txt", 'w') as f:
    for l in x_padded:
        line = "".join([str(x) + " " for x in l])
        f.write(line)
        f.write("\n")


with open("y_lyric.txt", 'w') as f:
    for l in y_padded:
        line = "".join([str(x) + " " for x in l])
        f.write(line)
        f.write("\n")

# with open()
# print(len(x_padded)) # 92639

# print(max(map(lambda x: len(x), x_idx)))
# print(min(map(lambda x: len(x), x_idx)))
# print(sum(map(lambda x: len(x), x_idx)) / len(x_idx))