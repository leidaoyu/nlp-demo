from utils.encode import build_vocab, text_to_indices

word2idx = {v.strip():k for k,v in  enumerate(open('vocab.txt', 'r', encoding='utf-8').readlines())}
sentence = '今天天气很好'
encode = text_to_indices(sentence, word2idx)
print(encode)