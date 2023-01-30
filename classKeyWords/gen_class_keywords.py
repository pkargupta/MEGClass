import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import tqdm

def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))

def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)

def main(args):
    static_repr_path = args.vocab_file
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)
        static_word_representations = vocab["static_word_representations"]
        vocab_words = vocab["vocab_words"]
        word_to_index = vocab["word_to_index"]

    with open(args.class_names, "r") as f:
            class_names = list(f.readlines())
            print("Class names: " + ', '.join(class_names))

    finished_class = set()
    masked_words = set(class_names)

    class_words = [[class_names[cls]] for cls in range(len(class_names))]
    class_words_representations = [[static_word_representations[word_to_index[class_names[cls]]]]
                                       for cls in range(len(class_names))]


    for t in range(1, args.T):
        class_representations = [average_with_harmonic_series(class_words_representation)
                                 for class_words_representation in class_words_representations]
        cosine_similarities = cosine_similarity_embeddings(static_word_representations,
                                                           class_representations)
        nearest_class = cosine_similarities.argmax(axis=1)
        similarities = cosine_similarities.max(axis=1)
        for cls in range(len(class_names)):
            if cls in finished_class:
                continue
            highest_similarity = -1.0
            highest_similarity_word_index = -1
            lowest_masked_words_similarity = 1.0
            existing_class_words = set(class_words[cls])
            stop_criterion = False
            for i, word in enumerate(vocab_words):
                if nearest_class[i] == cls:
                    if word not in masked_words:
                        if similarities[i] > highest_similarity:
                            highest_similarity = similarities[i]
                            highest_similarity_word_index = i
                    else:
                        if word not in existing_class_words:
                            stop_criterion = True
                            break
                        lowest_masked_words_similarity = min(lowest_masked_words_similarity, similarities[i])
                else:
                    if word in existing_class_words:
                        stop_criterion = True
                        break
            # the topmost t words are no longer the t words in class_words
            if lowest_masked_words_similarity < highest_similarity:
                stop_criterion = True

            if stop_criterion:
                finished_class.add(cls)
                class_words[cls] = class_words[cls][:-1]
                class_words_representations[cls] = class_words_representations[cls][:-1]
                break
            class_words[cls].append(vocab_words[highest_similarity_word_index])
            class_words_representations[cls].append(static_word_representations[highest_similarity_word_index])
            masked_words.add(vocab_words[highest_similarity_word_index])
        if len(finished_class) == len(class_names):
            break
    with open(args.outFile, "w") as f:
        for x in range(len(class_names)):
            f.write (', '.join(class_words[x]) + '\n')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outFile", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--T", type=int, default=100)

    args = parser.parse_args()
    print(vars(args))
    main(args)
