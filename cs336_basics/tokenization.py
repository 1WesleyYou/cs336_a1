import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

tmp = re.findall(PAT, "some text tht i'll pre-tokenize")
print(tmp)


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a given input file, returning a vocabulary mapping token IDs to byte strings and a list of merges.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    return vocab, merges
