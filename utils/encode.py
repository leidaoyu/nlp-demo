# 构建词表
def build_vocab(corpus):
    words = []
    for text in corpus:
        words.extend(list(text))
    vocab = list(set(words))
    vocab = sorted(vocab)
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab  # 添加特殊token
    word2idx = {w:i for i,w in enumerate(vocab)}
    # 保存词表到vocab.txt
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')
    return vocab, word2idx


# 数据预处理（转换为索引）
def text_to_indices(text, word2idx):
    return [word2idx['<bos>']] + [word2idx.get(w, word2idx['<unk>']) for w in list(text)] + [word2idx['<eos>']]