import gzip
import ftfy
import regex as re
import html
import os
from os.path import dirname, join

# Below implementation from Keras-cv clip_tokenizer.py
# https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/clip_tokenizer.py


def bytes_to_unicode():
    """Return a list of utf-8 bytes and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Variables init.
MAX_PROMPT_LENGTH = 77
filename = join(dirname(__file__), "bpe_simple_vocab_16e6.txt.gz")
merges = gzip.open(filename).read().decode("utf-8").split("\n")
byte_encoder = bytes_to_unicode()
merges = merges[1 : 49152 - 256 - 2 + 1]
merges = [tuple(merge.split()) for merge in merges]
vocab = list(bytes_to_unicode().values())
#print(vocab)
vocab = vocab + [v + "</w>" for v in vocab]
for merge in merges:
    vocab.append("".join(merge))
#print(vocab)
vocab.extend(["<|startoftext|>", "<|endoftext|>"])
#print(vocab)
encoder = dict(zip(vocab, range(len(vocab))))
bpe_ranks = dict(zip(merges, range(len(merges))))
#print(bpe_ranks)
cache = {
    "<|startoftext|>": "<|startoftext|>",
    "<|endoftext|>": "<|endoftext|>",
}
pat = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
    re.IGNORECASE,
)

def bpe(token):
    if token in cache:
        return cache[token]
    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    pairs = get_pairs(word)

    if not pairs:
        return token + "</w>"

    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float("inf")))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = " ".join(word)
    cache[token] = word
    return word

def encodeText(string_to_encode):
  bpe_tokens = []
  text = whitespace_clean(basic_clean(string_to_encode)).lower()
  print(text)
  for token in re.findall(pat, text):
      print()
      print(token)
      token = "".join(byte_encoder[b] for b in token.encode("utf-8"))
      print(token)
      bpe_tokens.extend(
          encoder[bpe_token] for bpe_token in bpe(token).split(" ")
      )
  result = [49406] + bpe_tokens + [49407]

  return result + [49407] * (MAX_PROMPT_LENGTH - len(result))
