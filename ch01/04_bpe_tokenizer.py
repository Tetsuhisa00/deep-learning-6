import importlib

bpe_train = importlib.import_module("03_bpe_train")
merge = bpe_train.merge


class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # IDからバイト列への対応表（0~255を登録）
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # マージされたトークンは元のトークンのバイト列を連結
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # 語彙サイズを設定
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        # 学習時の順序でマージルールを適用
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids

    def decode(self, ids):
        # 各トークンIDを対応するバイト列に変換
        byte_list = [self.id_to_bytes[i] for i in ids]

        # すべてのバイト列を連結
        text_bytes = b"".join(byte_list)

        # バイト列をUTF-8テキストに変換
        text = text_bytes.decode("utf-8", errors="replace")
        return text

# 学習済みのマージルール
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# トークナイザを作成
tokenizer = BPETokenizer(merge_rules)

# テキストをエンコード
text = "Hello世界😁"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)