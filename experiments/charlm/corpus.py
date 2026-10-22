"""Corpus preparation and character vocabularies for the Desmos char-LM study.

Corpora:
  homer        Samuel Butler's English prose translations of Homer's Odyssey
               (Project Gutenberg #1727) and Iliad (#2199), already on this
               machine. Public domain. Gutenberg boilerplate, the Odyssey preface
               and its footnotes are removed; whitespace collapses to one space.
  shakespeare  tiny-shakespeare (Karpathy's char-rnn repo, MIT licensed; the text
               is public-domain Shakespeare), downloaded with Santo's approval to
               data/tinyshakespeare/input.txt. Newlines are kept: the play
               format (speaker lines, blank-line breaks) is its structure.

Vocabularies (each belongs to one corpus):
  v32: homer, lowercase letters, space, and 5 punctuation marks (32 symbols, one
       32-wide matrix block in Desmos).
  v64: homer, case preserved, space, and 11 punctuation marks (64, two blocks).
  s32: shakespeare, lowercase letters, space, newline, and 4 punctuation marks.
Every source character must be either in a vocabulary, remapped, or dropped by an
explicit rule; anything else raises, so a new corpus cannot silently corrupt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Project Gutenberg plain-text files pg1727.txt (Odyssey) and pg2199.txt (Iliad) go here
DEFAULT_SOURCE_DIR = HERE / "data" / "gutenberg"
DEFAULT_SHAKESPEARE = HERE / "data" / "tinyshakespeare" / "input.txt"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
ODYSSEY_FILE = "pg1727.txt"
ILIAD_FILE = "pg2199.txt"

# Inline footnote markers in the Odyssey are digits glued to a word or closing
# punctuation, e.g. "abode.12" or "shirt15".
_FOOTNOTE_MARK = re.compile(r"(?<=[A-Za-z’”.,;:!?])\d+")
_WS = re.compile(r"\s+")
_SPACES = re.compile(r"[ \t]+")
_SPACE_AROUND_NL = re.compile(r" *\n *")
# Curly to straight quotes is typography, not information loss; counted apart.
_TYPOGRAPHIC = frozenset("\u201c\u201d\u2018\u2019")


@dataclass(frozen=True)
class Vocab:
    name: str
    symbols: str
    lowercase: bool
    remap: dict[str, str]  # source char -> replacement string ("" = drop)
    corpus: str
    sample_prompt: str

    @property
    def size(self) -> int:
        return len(self.symbols)

    def stoi(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.symbols)}


_DIGIT_DROP = dict.fromkeys(string.digits, "")

V32 = Vocab(
    name="v32",
    symbols=" " + string.ascii_lowercase + ".,;'\"",
    lowercase=True,
    remap={
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "?": ".", "!": ".",
        ":": ";",
        "—": " ", "–": " ", "-": " ",
        "(": " ", ")": " ", "[": " ", "]": " ",
        **_DIGIT_DROP,
    },
    corpus="homer",
    sample_prompt="then minerva said, ",
)

V64 = Vocab(
    name="v64",
    symbols=" " + string.ascii_lowercase + string.ascii_uppercase + ".,;:?!'\"-()",
    lowercase=False,
    remap={
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "—": "-", "–": "-",
        "[": "(", "]": ")",
        **_DIGIT_DROP,
    },
    corpus="homer",
    sample_prompt="Then Minerva said, ",
)

S32 = Vocab(
    name="s32",
    symbols=" \n" + string.ascii_lowercase + ",:.'",
    lowercase=True,
    remap={";": ",", "?": ".", "!": ".", "-": " ", "&": "", "$": "", **_DIGIT_DROP},
    corpus="shakespeare",
    sample_prompt="romeo:\n",
)

VOCABS = {v.name: v for v in (V32, V64, S32)}
for _vocab, _size in ((V32, 32), (V64, 64), (S32, 32)):
    if _vocab.size != _size:
        raise ValueError(f"{_vocab.name} must have {_size} symbols (part of the Desmos layout)")


def _slice_between(lines: list[str], start_pred, end_pred, start_after: int = 0) -> tuple[int, int]:
    start = next((i for i in range(start_after, len(lines)) if start_pred(lines[i])), None)
    if start is None:
        raise ValueError("start marker not found")
    end = next((i for i in range(start + 1, len(lines)) if end_pred(lines[i])), None)
    if end is None:
        raise ValueError("end marker not found")
    return start, end


def extract_odyssey(raw: str) -> str:
    lines = raw.split("\n")
    # The body starts at a bare "BOOK I" heading (the contents list uses "BOOK I.").
    start, end = _slice_between(
        lines,
        start_pred=lambda s: s.strip() == "BOOK I",
        end_pred=lambda s: s.strip() == "FOOTNOTES:",
    )
    return _FOOTNOTE_MARK.sub("", "\n".join(lines[start:end]))


def extract_iliad(raw: str) -> str:
    lines = raw.split("\n")
    contents, _ = _slice_between(
        lines,
        start_pred=lambda s: s.strip() == "BOOK I.",
        end_pred=lambda s: s.strip() == "BOOK I.",
    )
    start, end = _slice_between(
        lines,
        start_pred=lambda s: s.strip() == "BOOK I.",
        end_pred=lambda s: "END OF THE PROJECT GUTENBERG" in s,
        start_after=contents + 1,
    )
    return "\n".join(lines[start:end])


def load_clean_text(source_dir: Path) -> str:
    """Odyssey then Iliad, with all whitespace runs collapsed to one space."""
    odyssey = extract_odyssey((source_dir / ODYSSEY_FILE).read_text(encoding="utf-8"))
    iliad = extract_iliad((source_dir / ILIAD_FILE).read_text(encoding="utf-8"))
    return _WS.sub(" ", odyssey + " " + iliad).strip()


def map_text(text: str, vocab: Vocab) -> tuple[str, dict[str, int]]:
    """Map clean text into the vocab. Returns mapped text and per-category counts."""
    allowed = set(vocab.symbols)
    out: list[str] = []
    counts = {"exact": 0, "case_folded": 0, "quote_normalized": 0, "remapped": 0, "dropped": 0}
    unknown: set[str] = set()
    for ch in text:
        c = ch.lower() if vocab.lowercase else ch
        if c in allowed:
            out.append(c)
            counts["exact" if c == ch else "case_folded"] += 1
        elif ch in vocab.remap:
            rep = vocab.remap[ch]
            out.append(rep)
            if not rep:
                counts["dropped"] += 1
            elif ch in _TYPOGRAPHIC:
                counts["quote_normalized"] += 1
            else:
                counts["remapped"] += 1
        else:
            unknown.add(ch)
    if unknown:
        raise ValueError(f"characters with no rule in {vocab.name}: {sorted(unknown)!r}")
    if "\n" in vocab.symbols:
        mapped = _SPACE_AROUND_NL.sub("\n", _SPACES.sub(" ", "".join(out))).strip()
    else:
        mapped = _WS.sub(" ", "".join(out)).strip()
    return mapped, counts


def encode(text: str, vocab: Vocab) -> list[int]:
    table = vocab.stoi()
    missing = set(text) - set(table)
    if missing:
        raise ValueError(f"text contains symbols outside {vocab.name}: {sorted(missing)!r}")
    return [table[ch] for ch in text]


def decode(ids, vocab: Vocab) -> str:
    return "".join(vocab.symbols[int(i)] for i in ids)


def split_bounds(n: int, val_frac: float = 0.05, test_frac: float = 0.05) -> tuple[int, int]:
    """Contiguous split: train | val | test. Returns (train_end, val_end)."""
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("bad split fractions")
    # Exact rationals: in floats 1 - 0.05 - 0.05 is 0.8999999999999999, which put
    # the boundary one char early whenever n * 0.9 is an integer.
    val, test = Fraction(str(val_frac)), Fraction(str(test_frac))
    train_end = int(n * (1 - val - test))
    val_end = int(n * (1 - test))
    return train_end, val_end


def data_path(vocab_name: str) -> Path:
    vocab = VOCABS[vocab_name]
    return HERE / "data" / f"{vocab.corpus}_{vocab.name}.txt"


def load_shakespeare(path: Path) -> str:
    """Read the downloaded file as plain text after checking it is the known file."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHAKESPEARE_SHA256:
        raise ValueError(f"{path} sha256 {digest} does not match the pinned tiny-shakespeare file")
    return raw.decode("ascii")


def load_mapped(vocab_name: str) -> str:
    path = data_path(vocab_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python corpus.py` first")
    return path.read_text(encoding="ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--shakespeare", type=Path, default=DEFAULT_SHAKESPEARE)
    args = ap.parse_args()

    sources = {"homer": load_clean_text(args.source_dir)}
    if args.shakespeare.exists():
        sources["shakespeare"] = load_shakespeare(args.shakespeare)
    else:
        print(f"NOTICE: {args.shakespeare} not found; skipping the shakespeare corpus (source: {SHAKESPEARE_URL})")
    report: dict[str, object] = {
        "corpora": {
            "homer": "Homer, Odyssey (PG #1727) + Iliad (PG #2199), Samuel Butler prose translations",
            "shakespeare": f"tiny-shakespeare, {SHAKESPEARE_URL}, sha256 {SHAKESPEARE_SHA256}",
        },
        "source_chars": {k: len(v) for k, v in sources.items()},
        "source_distinct_chars": {k: len(set(v)) for k, v in sources.items()},
        "vocabs": {},
    }
    for vocab in VOCABS.values():
        if vocab.corpus not in sources:
            continue
        mapped, counts = map_text(sources[vocab.corpus], vocab)
        path = data_path(vocab.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mapped, encoding="ascii")
        total = sum(counts.values())
        report["vocabs"][vocab.name] = {
            "corpus": vocab.corpus,
            "symbols": vocab.symbols,
            "size": vocab.size,
            "mapped_chars": len(mapped),
            "sha256": hashlib.sha256(mapped.encode("ascii")).hexdigest(),
            "counts": counts,
            "pct": {k: round(100 * v / total, 4) for k, v in counts.items()},
        }
    out = HERE / "results" / "vocab_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
