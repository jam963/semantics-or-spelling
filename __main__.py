#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import string
from transformers import (
    AutoTokenizer,
    AutoModel
)
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm
from typing import Tuple, Optional


MODELS = [
     "roberta-large",
     "roberta-base",
     "gpt2-xl",
     "gpt2-large",
     "gpt2-medium",
     "gpt2",
     "bert-large-cased",
     "bert-base-cased",
     "xlnet-large-cased",
     "xlnet-base-cased",
     "bigscience/bloom-560m",
]

RNG = np.random.default_rng(1234567890)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sort_seqs(text: str, seq_len: int) -> Tuple[list, list]:
    """
    Returns all sequences with length seq_len, as well as vocabs of these sequences.
    """
    word_list = text.split()
    length = len(word_list)

    sequences = []
    vocabs = []
    for i in range(0, length - seq_len + 1, seq_len):
        sequence = word_list[i : i + seq_len]
        vocab = set(sequence)
        sequences.append(sequence)
        vocabs.append(vocab)

    return sequences, vocabs


def single_replace(word: str) -> Tuple[int, Optional[str]]:
    """
    Replaces a single character in a valid English word with a random character of the
    same case, returning the index of the replaced character and the edited string.
    """
    word_list = " ".join(word).split()
    char_types = [
        set(" ".join(string.ascii_lowercase).split()),
        set(" ".join(string.ascii_uppercase).split()),
    ]
    ix = RNG.integers(len(word))
    for cc in char_types:
        if word_list[ix] in cc:
            replacement = RNG.choice(list(cc - set([word[ix]])))
            word_list[ix] = replacement
            return (ix, "".join(word_list))
    return (ix, None)


def tokens_to_strings(word, tokenizer):
    """
    Converts tokens to string representations of tokens.
    """
    return np.array(
        [tokenizer.decode(x, clean_up_tokenization_spaces=False) for x in word]
    )


def map_context(text: str, context_length: int = 100) -> dict:
    """
    Takes text and returns sequences of equal context_length (in words),
    vocabs for each sequence, and a dict mapping each word to a sequence containing that word.
    Note that multiple words may be mapped to the same sequence.
    """
    context_seqs, context_vocabs = sort_seqs(text, context_length)
    context_map = {}
    for i, vv in enumerate(context_vocabs):
        for word in vv:
            context_map.update({word: context_seqs[i]})
    return context_map


def build_inputs(vocab: list, text: str) -> None:
    """Builds a csv file of valid English words, edited words, contexts, and edited contexts."""
    if not os.path.exists(os.path.join(ROOT_DIR, "inputs")):
        os.makedirs(os.path.join(ROOT_DIR, "inputs"))
        context_map = map_context(text)
        words = []
        edits = []
        edits_ix = []
        contexts = []
        edit_contexts = []
        for word in vocab:
            edit_ix, edit = single_replace(word)
            context_seq = context_map[word]
            edit_context = context_seq.copy()
            edit_context_ix = edit_context.index(word)
            edit_context[edit_context_ix] = edit
            if edit:
                words.append(word)
                edits.append(edit)
                edits_ix.append(edit_ix)
                contexts.append(" ".join(context_seq))
                edit_contexts.append(" ".join(edit_context))
        df = pd.DataFrame.from_dict(
            {
                "word": words,
                "edit": edits,
                "edit_ix": edits_ix,
                "context": contexts,
                "edit_context": edit_contexts,
            }
        ).dropna()
        df.to_csv(os.path.join(ROOT_DIR, "inputs/cleaned.csv"))


def main(vocab: list, text: str) -> None:
    if not os.path.exists(os.path.join(ROOT_DIR, "inputs", "cleaned.csv")):
        build_inputs(vocab, text)

    vocab_df = pd.read_csv(
        os.path.join(ROOT_DIR, "inputs", "cleaned.csv"), na_filter=False
    ).dropna()
    print(f"Final vocab size: {len(vocab_df)}")

    words = vocab_df["word"].astype(str).tolist()
    edits = vocab_df["edit"].astype(str).tolist()
    contexts = vocab_df["context"].astype(str).tolist()
    edit_contexts = vocab_df["edit_context"].astype(str).tolist()

    if not os.path.exists(os.path.join(ROOT_DIR, "outputs")):
        os.makedirs(os.path.join(ROOT_DIR, "outputs"))
        models_left = MODELS
    else:
        # Don't forget naming convention here! Replace / with space when saving too.
        outputs = [
            oo.replace(".csv", "").replace(" ", "/")
            for oo in os.listdir(os.path.join(ROOT_DIR, "outputs"))
        ]
        models_left = list(set(MODELS) - set(outputs))

    for m in models_left:
        print(f"Starting {m}...")

        model_data = {
            "word_token_length": [],
            "edit_token_length": [],
            "word_token_str": [],
            "edit_token_str": [],
            "model": [m] * len(words),
            "raw_distance": [],
            "raw_context_distance": [],
            "spearman": [],
            "context_spearman": []
        }
        model = AutoModel.from_pretrained(m).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(m, add_prefix_space=True)
        with torch.no_grad():
            for word, edit, context, edit_context in tqdm(
                zip(words, edits, contexts, edit_contexts)
            ):
                # Encodings
                word_encoding = tokenizer(
                    word, add_special_tokens=False, return_tensors="pt"
                )
                word_token_length = len(word_encoding["input_ids"][0])
                word_token_str = tokens_to_strings(
                    word_encoding["input_ids"][0], tokenizer
                )

                edit_encoding = tokenizer(
                    edit, add_special_tokens=False, return_tensors="pt"
                )
                edit_token_length = len(edit_encoding["input_ids"][0])
                edit_token_str = tokens_to_strings(
                    edit_encoding["input_ids"][0], tokenizer
                )

                # Model hidden states, zscored
                word_output = torch.mean(
                        model(**word_encoding.to("cuda")).last_hidden_state[0], dim=0
                    ).cpu().numpy()

                edit_output = torch.mean(
                        model(**edit_encoding.to("cuda")).last_hidden_state[0], dim=0
                    ).cpu().numpy()

                raw_distance = cosine(word_output, edit_output)
                spearman = spearmanr(word_output, edit_output)
                # Update model_data with non-contextual data
                model_data["word_token_length"].append(word_token_length)
                model_data["edit_token_length"].append(edit_token_length)
                model_data["word_token_str"].append(word_token_str)
                model_data["edit_token_str"].append(edit_token_str)
                model_data["raw_distance"].append(raw_distance)
                model_data["spearman"].append(spearman)

                # Context encodings
                context_encoding = tokenizer(
                    context.split(), is_split_into_words=True, return_tensors="pt"
                )
                edit_context_encoding = tokenizer(
                    edit_context.split(), is_split_into_words=True, return_tensors="pt"
                )
                context_span = context_encoding.word_to_tokens(
                    0, context.split().index(word)
                )
                edit_context_span = edit_context_encoding.word_to_tokens(
                    0, edit_context.split().index(edit)
                )

                # Contextual model outputs
                context_output = torch.mean(
                        model(**context_encoding.to("cuda")).last_hidden_state[0][
                            context_span.start:context_span.end
                        ],
                        dim=0,
                    ).cpu().numpy()

                edit_context_output = torch.mean(
                        model(**edit_context_encoding.to("cuda")).last_hidden_state[0][
                            edit_context_span.start:edit_context_span.end
                        ],
                        dim=0,
                    ).cpu().numpy()

                raw_context_distance = cosine(context_output, edit_context_output)
                context_spearman = spearmanr(context_output, edit_context_output)
                # Update model data with contextual data
                model_data["context_spearman"].append(context_spearman)
                model_data["raw_context_distance"].append(raw_context_distance)

        output_df = pd.concat([vocab_df, pd.DataFrame.from_dict(model_data)], axis=1)
        output_df.to_csv(
            os.path.join(ROOT_DIR, "outputs", f'{m.replace("/", " ")}.csv')
        )


if __name__ == "__main__":
    # Get dataset
    text = " ".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    )
    # Restrict input vocab to all-alphanumeric
    vocab = list({word for word in text.split() if word.isalpha() and len(word) > 3})

    # Run main inference loop
    main(vocab, text)
