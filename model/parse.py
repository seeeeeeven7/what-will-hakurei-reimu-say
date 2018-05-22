data_ = ""
with open('../data/talk_in_game/all.txt', 'r', encoding = 'utf8') as f:
    data_ += f.read()
data_ = data_.lower()
# Convert to 1-hot coding
vocab = sorted(list(set(data_)))
print(len(vocab))
print(vocab)