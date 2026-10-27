# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Byte-level BPE from scratch.

Training operates on WORD TYPES (unique pre-tokens weighted by frequency),
not on the raw stream — the standard trick that makes pure-Python BPE
feasible: a corpus has billions of tokens but only ~1M unique words.
Pair counts are updated incrementally (only words containing the merged
pair are touched) and the best pair comes from a lazy max-heap.

Encoding caches per-word token ids, so repeated words cost a dict lookup.
"""

import heapq
import json
import re
from collections import Counter

# GPT-2-style pre-tokenizer, simplified: contraction | space+letters |
# space+digits | space+punct run | whitespace. Leading space stays attached
# to the word so " cell" and "cell" are distinct tokens, encoding word position.
PRETOKEN = re.compile(r"'[a-z]{1,2}| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+")

ENDOFTEXT = "<|endoftext|>"


def pretokenize(text):
    return PRETOKEN.findall(text)


class BPE:
    def __init__(self, merges=None):
        self.merges = merges or []          # list of (id_a, id_b), rank = index
        self._rebuild()

    def _rebuild(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        # id -> raw bytes, for decoding
        self.id_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.id_bytes.append(self.id_bytes[a] + self.id_bytes[b])
        self.eot_id = 256 + len(self.merges)  # single special token, last id
        self.vocab_size = self.eot_id + 1
        self._cache = {}

    # ---------------- training ----------------
    @classmethod
    def train(cls, texts, vocab_size, log_every=1000):
        """texts: iterable of strings. vocab_size includes 256 bytes + 1 special."""
        n_merges = vocab_size - 256 - 1
        word_freq = Counter()
        for t in texts:
            word_freq.update(pretokenize(t))
        words = [tuple(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())
        print(f"word types: {len(words):,}, total words: {sum(freqs):,}")

        pair_counts = Counter()
        pair2words = {}
        for wi, word in enumerate(words):
            f = freqs[wi]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += f
                pair2words.setdefault(pair, set()).add(wi)

        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)
        merges = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg or -neg == 0:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)
            touched = set()
            for wi in list(pair2words.get(pair, ())):
                word, f = words[wi], freqs[wi]
                for p in zip(word, word[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                    s = pair2words.get(p)
                    if s is not None:
                        s.discard(wi)
                merged, j = [], 0
                while j < len(word):
                    if j + 1 < len(word) and (word[j], word[j + 1]) == pair:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(word[j])
                        j += 1
                words[wi] = word = tuple(merged)
                for p in zip(word, word[1:]):
                    pair_counts[p] += f
                    touched.add(p)
                    pair2words.setdefault(p, set()).add(wi)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            if len(merges) % log_every == 0:
                print(f"  merge {len(merges):,}/{n_merges:,}", flush=True)
        return cls(merges)

    # ---------------- encode / decode ----------------
    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        seq = list(word.encode("utf-8"))
        while len(seq) > 1:
            best, best_rank = None, None
            for pair in zip(seq, seq[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id, merged, j = 256 + best_rank, [], 0
            while j < len(seq):
                if j + 1 < len(seq) and (seq[j], seq[j + 1]) == best:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(seq[j])
                    j += 1
            seq = merged
        self._cache[word] = seq
        return seq

    def encode(self, text):
        out = []
        for w in pretokenize(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        return b"".join(
            self.id_bytes[i] for i in ids if i != self.eot_id
        ).decode("utf-8", errors="replace")

    # ---------------- persistence ----------------
    def save(self, path, meta=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "special": [ENDOFTEXT],
                       "meta": meta or {}}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([tuple(p) for p in data["merges"]])
