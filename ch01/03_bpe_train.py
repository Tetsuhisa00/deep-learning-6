from collections import defaultdict


def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


def train_bpe(text, vocab_size):
    # テキストを0~255のID列に変換
    ids = list(text.encode("utf-8"))

    # 1,マージ回数を決定
    num_merges = vocab_size - 256 # 256は初期の語彙サイズ
    merge_rules = {}

    for step in range(num_merges):
        # 隣接ペアの統計を取得
        counts = count_pairs(ids)

        # 2,ペアが存在しない場合の処理
        if not counts:
            break

        # 3,最頻出ペアを選択
        best_pair = max(counts, key=counts.get)

        # 4, 新しいトークンIDを割り当て
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # マージを実行
        ids = merge(ids, best_pair, new_id)
    return merge_rules


if __name__ == "__main__":
    text = "Hello world! This is BPE training."

    # BPEを学習
    merge_rules = train_bpe(text, vocab_size=260)
    print(merge_rules)