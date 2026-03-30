import regex as re
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# tmp = re.findall(PAT, "some text tht i'll pre-tokenize")
# print(tmp)


def bpe_train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a given input file, returning a vocabulary mapping token IDs to byte strings and a list of merges.
    """
    with open(input_path, "rb") as f:
        num_processes = 4
        # TODO: make this special tokens to multi version one
        boundaries = find_chunk_boundaries(f, num_processes,
                                           bytes(special_tokens[0], "utf-8"))
                                           

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    return vocab, merges


def main():
    input_path = "cs336_basics/test.txt"
    size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = bpe_train(input_path, size, special_tokens)


if __name__ == "__main__":
    main()
