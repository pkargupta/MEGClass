# MEGClass: Text Classification with Extremely Weak Supervision via Mutually-Enhancing Text Granularities

## Setup
We use python=3.8, torch=1.13.1, cudatoolkit=11.3, and a single NVIDIA RTX A6000 GPU. Other packages can be installed using:
```
pip install -r requirements.txt
```

Specify the variables `DATA_FOLDER_PATH` and `INTERMEDIATE_DATA_FOLDER_PATH` within `utils.py`. `DATA_FOLDER_PATH` should be where your datasets are saved (all provided within the `datasets/` folder) and `INTERMEDIATE_DATA_FOLDER_PATH` is where all of the intermediate data is stored (e.g. pickle files for class-oriented sentence and class representations, where the final pseudo-training dataset is stored).

## Training
In order to learn the contextualized sentence and document representations for a specific dataset (in this case, 20News), run the following command:

```
time CUDA_VISIBLE_DEVICES=[gpu] python run.py --gpu [gpu] --dataset_name 20News
```
### Arguments
The following are the primary arguments for MEGClass

- `dataset_name`
- `gpu` $\rightarrow$ GPU to use; refer to nvidia-smi
- `emb_dim` $\rightarrow$ default=768; Sentence and document embedding dimensions (default based on bert-base-uncased).
- `num_heads` $\rightarrow$ default=2; Number of heads to use for MultiHeadAttention.
- `batch_size` $\rightarrow$ default=64; Batch size of documents.
- `epochs` $\rightarrow$ default=4; Number of epochs to learn contextualized representations for during single iteration.
- `max_sent` $\rightarrow$ default=150; For padding, the max number of sentences within a document.
- `temp` $\rightarrow$ default=0.1; Temperature scaling factor; regularization.
- `lr` $\rightarrow$ default=1e-3, Learning rate for training contextualized embeddings.
- `iters` $\rightarrow$ default=1; Number of iterations of iterative feedback.
- `k` $\rightarrow$ default=0.075; Top k proportion of docs added to each class set (7.5%).
- `doc_thresh` $\rightarrow$ default=0.5; Pseudo-training dataset threshold.
- `pca` $\rightarrow$ default=64; Number of dimensions projected to in PCA, -1 means not doing PCA.
