data = open('../data/TH06.txt', 'r', encoding = 'utf8').read()
chars = list(set(data))
VOCAB_SIZE = len(chars)
print(VOCAB_SIZE)