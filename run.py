import subprocess
import os
import pickle as pk
import argparse
import time
import numpy as np
import megclass, train_text_classifier, train_soft_classifier
import class_oriented_sent_representations
import static_representations
from utils import (DATA_FOLDER_PATH, INTERMEDIATE_DATA_FOLDER_PATH)

def main(args):
    # initialize representations before iterative process: 

    print("Starting to compute static representations...")
    static_representations.main(args)

    print("Starting to compute class-oriented document representations...")
    class_oriented_sent_representations.main(args)

    start = time.time()
    megclass.main(args)
    
    if args.soft:
        print("Training classifier with soft labels!")
        train_soft_classifier.main(args)
    else:
        print("Training classifier with hard labels!")
        train_text_classifier.main(args)

    print(f"Total Time: {(time.time()-start)/60} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # new
    parser.add_argument("--emb_dim", type=int, default=768, help="sentence and document embedding dimensions; all-roberta-large-v1 uses 1024.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads to use for MultiHeadAttention.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of documents.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train for.")
    parser.add_argument("--accum_steps", type=int, default=1, help="For training.")
    parser.add_argument("--max_sent", type=int, default=150, help="For padding, the max number of sentences within a document.")
    parser.add_argument("--temp", type=float, default=0.1, help="temperature scaling factor; regularization")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for training contextualized embeddings.")
    parser.add_argument("--iters", type=int, default=4, help="number of iters for re-training embeddings.")
    parser.add_argument("--k", type=float, default=0.075, help="Top k percent docs added to class set.")
    parser.add_argument(
            "--train_data_dir",
            default=INTERMEDIATE_DATA_FOLDER_PATH,
            type=str,
            help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
        )
    parser.add_argument(
        "--test_data_dir",
        default=DATA_FOLDER_PATH,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    # general args and static repr args
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=False, default="seeds.txt")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--emb", type=str, default='plm')
    parser.add_argument("--vocab_min_occurrence", type=int, default=5)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--soft", action="store_true", help="Whether to run use soft pseudo-labels for final fine-tuning.")

    # class oriented doc repr args
    parser.add_argument("--attention_mechanism", type=str, default="mixture")
    parser.add_argument("--do_sent", type=str, default="yes")
    parser.add_argument("--T", type=int, default=100)

    # doc-class alignment args
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, -1 means not doing PCA.")
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans"], default="gmm")
    parser.add_argument("--document_repr_type", default="mixture")
    parser.add_argument("--alignment", default="document_representations")
    parser.add_argument("--representation", default="plm")

    # prep text classifier dataset args
    parser.add_argument("--suffix", type=str, default="pca64.clusgmm.bbu-12.mixture.42")
    parser.add_argument("--confidence_threshold", type=float, default=2, help="Training data confidence threshold.")
    parser.add_argument("--doc_thresh", type=float, default=0.5, help="Pseudo-training dataset threshold.")
    parser.add_argument("--granularity", default="document", help="Select either \"sent\" or \"document\".")
    parser.add_argument("--repr", default="plm") # make this into a list of representation types


    # train text classifier args
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=DATA_FOLDER_PATH,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",  # roberta-base
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_suffix",
        default=None,
        type=str,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--test_suffix", default="", type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_false", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_false", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_false", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_false", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_false", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # iterative approach args

    parser.add_argument("--iter", type=int, default=0, help="Iteration # of updating GMM and expanding pseudo-training dataset.")
    parser.add_argument("--max_iters", type=int, default=4)
    parser.add_argument("--initgmm", type=int, default=0, help="0: Use all documents to initialize gmm during first iteration. 1: Use only confident documents.")

    # confidence comparison args
    parser.add_argument("--lower", type=float, default=0.3, help="Lower proportion threshold for sentences to be added with their documents.")
    parser.add_argument("--upper", type=float, default=0.8, help="Upper proportion threshold for documents to be added instead of their sentences.")
    parser.add_argument("--usegmmconf", type=int, default=0, help="0: use cosine sim for sentence confidence, 1: use gmm prediction prob for sentence confidence")
    parser.add_argument("--weights", type=str, default="5.0 3.0 2.0", help="weights used for weighted average of different confident metrics. 1: top gmm density, 2: second top gmm density, 3: cos sim of gmm predicted class.")

    args = parser.parse_args()

    args.suffix = f"pca{args.pca}.bbu-12.mixture.42"

    if args.train_suffix is None:
        args.train_suffix = f"pca{args.pca}.bbu-12.mixture.42"


    if args.output_dir is None:
        args.output_dir = f"../models/{args.dataset_name}/{args.model_name_or_path}_{args.train_suffix}_{args.representation}"

    # default arg values based on original bash script
    args.max_seq_length = 512
    args.per_gpu_train_batch_size = 32
    args.per_gpu_eval_batch_size = 32
    args.logging_steps = 100000
    args.save_steps = -1

    print(vars(args))
    main(args)
