"""Needle-in-a-haystack: building the samples and scoring them.

One design choice worth stating, because it sets what the probe can mean. The models here
are base LMs, not instruction-tuned ones. Asked a question in an instruction format, a
base model can fail because it does not follow the instruction, and the score then
measures instruction-following instead of recall, which would push the unquantized
baseline down and make every quantized arm look better by comparison.

So the prompt ends mid-sentence, at "The secret code for X is", and the model only has to
copy a four-digit number it has already seen. A miss can then only mean the state no
longer holds the number.
"""
import random

# Invented words, so that no answer can be guessed from pretraining
NEEDLE_WORDS = [
    "zorblax", "quenthil", "vremdak", "plithorn", "gaxeltu",
    "brindow", "kelvoth", "mirquen", "tazzelin", "wobrunth",
]

NEEDLE_TMPL = "\n\nThe secret code for {word} is {code}.\n\n"
SUFFIX_TMPL = (
    "\n\nQuestion: What is the secret code for {word}?\n"
    "Answer: The secret code for {word} is"
)


def make_sample(tok, filler_ids, total_len, depth, word, code):
    """Insert the needle at relative position depth and return prompt ids of ~total_len.

    depth runs 0 to 1. At 0 the needle is as early as it can be, so it sits in the state
    across the most decode steps; at 1 it is immediately before the question.
    """
    needle_ids = tok(NEEDLE_TMPL.format(word=word, code=code),
                     add_special_tokens=False).input_ids
    suffix_ids = tok(SUFFIX_TMPL.format(word=word),
                     add_special_tokens=False).input_ids

    budget = total_len - len(needle_ids) - len(suffix_ids)
    if budget < 32:
        raise ValueError(f"total_len={total_len} leaves no room for needle + question")

    body = list(filler_ids[:budget])
    cut = int(len(body) * depth)
    prompt_ids = body[:cut] + needle_ids + body[cut:] + suffix_ids
    return prompt_ids


def score(generated_text, code):
    """A hit is the digits appearing anywhere in the continuation; formatting is not
    penalised, since the question is about the state and not about presentation."""
    return str(code) in generated_text


def sample_spec(rng: random.Random):
    """One (word, four-digit code) pair. Years are skipped so the code cannot be a
    plausible completion on its own."""
    word = rng.choice(NEEDLE_WORDS)
    code = rng.randint(1000, 9999)
    while 1900 <= code <= 2100:
        code = rng.randint(1000, 9999)
    return word, code
