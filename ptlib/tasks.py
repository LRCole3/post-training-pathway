"""A small synthetic 'world' for the language-model notebooks (01-05).

Pretraining text mixes arithmetic facts written as prose with filler sentences. Instruction
data uses a Q/A chat format that never appears in pretraining. Demonstrations come from two
kinds of annotator ("careful" and "sloppy"), so the SFT data contains realistic mistakes, and
a programmatic verifier gives the ground-truth ('gold') reward that real labs never have.
"""
import re
import numpy as np

# ---------------------------------------------------------------- vocabulary (character level)
PAD = "\0"
CHARS = PAD + "\n" + " .,?!:+=-" + "0123456789" + "abcdefghijklmnopqrstuvwxyz" + "AQ"
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
PAD_ID, EOS_ID = STOI[PAD], STOI["\n"]          # a response ends at its first newline


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(CHARS[i] for i in ids if i != PAD_ID)


# ---------------------------------------------------------------- held-out problems
_rng = np.random.default_rng(1234)
_all2 = [(a, b) for a in range(10, 100) for b in range(10, 100)]
_idx2 = _rng.permutation(len(_all2))
TEST_2D = [_all2[i] for i in _idx2[:400]]                    # never seen in any training data
TRAIN_2D = [_all2[i] for i in _idx2[400:]]
_TEST2_SET = set(TEST_2D)
TEST_3D = [tuple(x) for x in _rng.integers(100, 1000, size=(400, 2))]
_TEST3_SET = set(TEST_3D)


def sample_pairs(n, digits=2, rng=None, split="train"):
    """Random (a, b) operand pairs. split='train' excludes the held-out test pairs."""
    rng = rng or np.random.default_rng()
    if split == "test":
        pool = TEST_2D if digits == 2 else TEST_3D
        return [pool[i] for i in rng.integers(0, len(pool), n)]
    if digits == 2:
        return [TRAIN_2D[i] for i in rng.integers(0, len(TRAIN_2D), n)]
    out = []
    while len(out) < n:
        a, b = (int(x) for x in rng.integers(100, 1000, 2))
        if (a, b) not in _TEST3_SET:
            out.append((a, b))
    return out


# ---------------------------------------------------------------- pretraining corpus
_ADJ = "red big small old new happy quiet bright dark green".split()
_NOUN = "fox dog cat bird tree house river child farmer robot".split()
_VERB = "sees likes finds follows helps watches".split()
_ITEM = "apples pears books coins stones birds cars".split()


def _prose(rng):
    k = rng.integers(0, 4)
    if k == 0:
        return f"the {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} the {rng.choice(_ADJ)} {rng.choice(_NOUN)}."
    if k == 1:
        return f"there are {rng.integers(2, 60)} {rng.choice(_ITEM)} near the {rng.choice(_NOUN)}."
    if k == 2:
        s = int(rng.integers(1, 90))
        return "she counted " + " ".join(str(s + i) for i in range(int(rng.integers(3, 7)))) + "."
    return f"a {rng.choice(_NOUN)} and a {rng.choice(_NOUN)} went to the {rng.choice(_ADJ)} river."


def _fact(a, b, rng):
    c = a + b
    k = rng.integers(0, 3)
    return [f"{a}+{b}={c}", f"{a} plus {b} is {c}.", f"the sum of {a} and {b} is {c}."][k]


def pretraining_corpus(n_lines=60000, frac_2d=0.45, frac_3d=0.08, seed=0):
    """One long string of newline-separated lines. No line uses the Q/A chat format."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n_lines):
        u = rng.random()
        if u < frac_2d:
            lines.append(_fact(*sample_pairs(1, 2, rng)[0], rng))
        elif u < frac_2d + frac_3d:
            lines.append(_fact(*sample_pairs(1, 3, rng)[0], rng))
        else:
            lines.append(_prose(rng))
    return "\n".join(lines) + "\n"


def prose_corpus(n_lines=3000, seed=99):
    """Held-out prose only: used to measure forgetting of general 'language' after fine-tuning."""
    rng = np.random.default_rng(seed)
    return "\n".join(_prose(rng) for _ in range(n_lines)) + "\n"


# ---------------------------------------------------------------- chat format
_TEMPLATES = ["Q: what is {a} plus {b}?\nA:", "Q: add {a} and {b}.\nA:", "Q: {a}+{b}=?\nA:"]


def make_prompt(a, b, template=None, rng=None):
    rng = rng or np.random.default_rng()
    t = _TEMPLATES[int(rng.integers(0, len(_TEMPLATES)))] if template is None else _TEMPLATES[template]
    return t.format(a=a, b=b)


def format_answer(c, style):
    return f" {c}\n" if style == "terse" else f" the answer is {c}.\n"


def _mistake(a, b, rng):
    """Plausible human arithmetic errors: dropped carry, off by ten, off by one."""
    c = a + b
    k = rng.integers(0, 3)
    if k == 0:   # add digit-wise without carrying
        s, p, x, y = 0, 1, a, b
        while x or y:
            s += ((x % 10 + y % 10) % 10) * p; x //= 10; y //= 10; p *= 10
        if s != c:
            return s
    if k == 1:
        return c + int(rng.choice([-10, 10]))
    return c + int(rng.choice([-1, 1]))


def demonstrations(n, digits=2, p_sloppy=0.3, sloppy_error=0.6, seed=0):
    """Instruction-tuning data written by a mix of careful and sloppy annotators.

    careful: always correct, writes a full sentence 80% of the time.
    sloppy : wrong `sloppy_error` of the time, terse 80% of the time.
    Returns a list of dicts with prompt, response, a, b, correct, style, annotator.
    """
    rng = np.random.default_rng(seed)
    out = []
    for a, b in sample_pairs(n, digits, rng):
        sloppy = rng.random() < p_sloppy
        if sloppy:
            style = "terse" if rng.random() < 0.8 else "sentence"
            wrong = rng.random() < sloppy_error
        else:
            style = "sentence" if rng.random() < 0.8 else "terse"
            wrong = False
        c = _mistake(a, b, rng) if wrong else a + b
        out.append(dict(prompt=make_prompt(a, b, rng=rng), response=format_answer(c, style), a=a, b=b,
                        correct=not wrong, style=style, annotator="sloppy" if sloppy else "careful"))
    return out


# ---------------------------------------------------------------- verifier (the 'gold' reward)
_TERSE = re.compile(r"^ (\d+)$")
_SENT = re.compile(r"^ the answer is (\d+)\.$")


def parse_response(text):
    """Return (answer or None, style or None) for the first line of a response."""
    line = text.split("\n", 1)[0]
    m = _TERSE.match(line)
    if m:
        return int(m.group(1)), "terse"
    m = _SENT.match(line)
    if m:
        return int(m.group(1)), "sentence"
    return None, None


def gold_reward(a, b, text):
    """1 if the response is well formed and correct, else 0. Style does not matter."""
    ans, _ = parse_response(text)
    return float(ans == a + b)


def a_b_from_prompt(prompt):
    nums = [int(x) for x in re.findall(r"\d+", prompt)]
    return nums[0], nums[1]
