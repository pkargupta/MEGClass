set -e

gpu=$1
dataset=$2
vocab=$3
outfile=$4

CUDA_VISIBLE_DEVICES=${gpu} python static_repr.py --dataset_name ${dataset} --vocab_file ${vocab}
CUDA_VISIBLE_DEVICES=${gpu} python gen_class_keywords.py --dataset_name ${dataset} --vocab_file ${vocab} --out_file ${outfile}
