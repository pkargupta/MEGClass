import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import pickle as pk
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import DATA_FOLDER_PATH, INTERMEDIATE_DATA_FOLDER_PATH, tensor_to_numpy, docToClass, getTargetClassSet
from utils import docToClassSet, getSentClassRepr, evaluate_predictions, getDSMapAndGold
from utils import generateDataset, updateClassSet
from model import bertSentenceEmb, MEGClassModel

def weighted_class_contrastive_loss(args, sample_outputs, class_weights, class_embds):
    # k: B x C, class_weights: B x C
    numerator = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[:,None], class_embds, axis=2)/args.temp)
    denom = numerator.sum(dim=1).unsqueeze(-1)
    weighted_loss = -1 * (torch.log(numerator/(denom)) * class_weights).sum() # B x C -> B
    return weighted_loss/len(sample_outputs)

# train multi-head self-attention network -> learn contextualized representations
def contextEmb(args, sent_representations, mask, class_repr, class_weights, 
                doc_lengths, new_data_path, device):
    sent_representations = torch.from_numpy(sent_representations)
    mask = torch.from_numpy(mask).to(torch.bool)
    class_weights = torch.from_numpy(class_weights)
    classes = torch.from_numpy(class_repr).float().to(device)
    dataset = TensorDataset(sent_representations, mask, class_weights)
    sampler = SequentialSampler(dataset)
    dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, shuffle=False)
    # sent_representations: N docs x L sentences x 768 emb (L with padding is always max_sents=50)
    model = MEGClassModel(args.emb_dim, args.emb_dim, args.num_heads).to(device)

    total_steps = len(dataset_loader) * args.epochs / args.accum_steps
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

    print("Starting to train!")

    for i in tqdm(range(args.epochs)):
        total_train_loss = 0

        for batch in tqdm(dataset_loader):
            model.train()
            input_emb = batch[0].to(device).float()
            input_mask = batch[1].to(device)
            input_weights = batch[2].to(device).float()
            
            c_doc, _, alpha, c_sent = model(input_emb, mask=input_mask)
            c_doc = c_doc.squeeze(1)

            loss = weighted_class_contrastive_loss(args, c_doc, input_weights, classes) / args.accum_steps

            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_train_loss = total_train_loss / len(dataset_loader)
        print(f"Average training loss: {avg_train_loss}")

    model.eval()

    torch.save(model.state_dict(), os.path.join(new_data_path, f"{args.dataset_name}_model_e{args.epochs}.pth"))

    print("Starting to evaluate!")

    evalsampler = SequentialSampler(dataset)
    eval_loader = DataLoader(dataset, sampler=evalsampler, batch_size=args.batch_size, shuffle=False)

    doc_predictions = None
    attention_weights = np.zeros_like(mask, dtype=float)
    updated_sent_repr = np.zeros_like(sent_representations)
    final_doc_emb = np.zeros((len(class_weights), args.emb_dim))
    idx = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_emb = batch[0].to(device).float()
            input_mask = batch[1].to(device)

            c_doc, _, alpha, c_sent = model(input_emb, mask=input_mask)
            c_doc = c_doc.squeeze(1)
            c_sent, c_doc, alpha = tensor_to_numpy(c_sent), tensor_to_numpy(c_doc), tensor_to_numpy(torch.squeeze(alpha, dim=2))

            final_doc_emb[idx:idx+c_doc.shape[0], :] = c_doc
            attention_weights[idx:idx+c_doc.shape[0], :] = alpha
            updated_sent_repr[idx:idx+c_doc.shape[0], :, :] = c_sent

            idx += c_doc.shape[0]

            doc_class = docToClass(c_doc, class_repr)
            if doc_predictions is None:
                doc_predictions = doc_class
            else:
                doc_predictions = np.append(doc_predictions, doc_class)

    return doc_predictions, final_doc_emb, updated_sent_repr, attention_weights


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data_path = os.path.join(DATA_FOLDER_PATH, args.dataset_name)
    new_data_path = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # this is where we save all representations
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # read in class-oriented sentence + class representations and labels
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name, "dataset.pk"), "rb") as f:
        dataset = pk.load(f)
        sent_dict = dataset["sent_data"]
        cleaned_text = dataset["cleaned_text"]
        class_names = np.array(dataset["class_names"])

    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name, "document_repr_lm-bbu-12-mixture-plm.pk"), "rb") as f:
        reprpickle = pk.load(f)
        class_words = reprpickle["class_words"]

    gold_labels, gold_sent_labels, doc_to_sent = getDSMapAndGold(args, sent_dict)
    num_classes = len(class_words)
    repr_out = getSentClassRepr(args)
    if len(repr_out) == 2:
        sent_repr, class_repr = repr_out
    else:
        doc_repr, sent_repr, class_repr = repr_out

    # initialize class sets to word-based class representations and compute target class distribution
    init_class_set = [np.array([class_repr[i]]) for i in np.arange(len(class_repr))] # C x CD x E
    class_set = [np.array([class_repr[i]]) for i in np.arange(len(class_repr))] # C x CD x E
    padded_sent_repr, doc_lengths, sentence_mask = bertSentenceEmb(args, doc_to_sent, sent_repr)
    init_class_weights, init_sent_weights = getTargetClassSet(padded_sent_repr, doc_lengths, class_set, alpha=None, set_weights=None)
    print("Initial Target Class Distribution Evaluation:")
    evaluate_predictions(gold_labels, np.argmax(init_class_weights, axis=1))

    # learn contextualized reprs + update class set for each iteration
    curr_class_weights = init_class_weights
    for i in np.arange(args.iters):
        print(f"Iter {i}: Training!")
        # train contextualized sent + doc embeddings
        doc_to_class, final_doc_emb, updated_sent_repr, updated_sent_weights = contextEmb(args, padded_sent_repr, sentence_mask, class_repr, curr_class_weights, doc_lengths, new_data_path, device)
        doc_pred = np.rint(doc_to_class)
        print(f"Iteration {i} Training Evaluation: ")
        evaluate_predictions(gold_labels, doc_pred)

        # pca transform on contextualized reprs
        _pca = PCA(n_components=args.pca, random_state=args.random_state)
        pca_doc_repr = _pca.fit_transform(final_doc_emb)
        pca_class_repr = [_pca.transform(init_class_set[i]) for i in np.arange(len(init_class_set))]
        doc_class_assignment, doc_class_cos = docToClassSet(pca_doc_repr, pca_class_repr, None)
        print(f"Iteration {i} PCA Evaluation: ")
        evaluate_predictions(gold_labels, doc_class_assignment)

        # soft pseudo-labels in case of soft classifier
        reg_temp = 0.2
        doc_class_dist = np.exp(doc_class_cos/reg_temp)/np.sum(np.exp(doc_class_cos/reg_temp), axis=1).reshape(-1,1)

        # update class set or generate final pseudo training dataset
        doc_rank = np.max(doc_class_cos, axis=1)
        if i == args.iters - 1:
            thresh = args.doc_thresh
            finalconf = generateDataset(doc_class_assignment, doc_rank, doc_class_dist, num_classes, cleaned_text, 
                                        gold_labels, data_path, new_data_path, thresh, write=True)
        else:
            thresh = args.k
            finalconf = generateDataset(doc_class_assignment, doc_rank, doc_class_dist, num_classes, cleaned_text, 
                                        gold_labels, data_path, new_data_path, thresh, write=False)
            updateClassSet(finalconf, final_doc_emb, class_set)

            # recompute target class distribution based on updated contextualized sent repr, sentence weights, and class set
            curr_class_weights = getTargetClassSet(updated_sent_repr, doc_lengths, class_set, alpha=updated_sent_weights, set_weights=None)
            evaluate_predictions(gold_labels, np.argmax(curr_class_weights, axis=1))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="agnews")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, -1 means not doing PCA.")
    parser.add_argument("--gpu", type=int, default=7, help="GPU to use; refer to nvidia-smi")
    parser.add_argument("--emb_dim", type=int, default=768, help="sentence and document embedding dimensions; all-roberta-large-v1 uses 1024.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads to use for MultiHeadAttention.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of documents.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train for.")
    parser.add_argument("--accum_steps", type=int, default=1, help="For training.")
    parser.add_argument("--max_sent", type=int, default=150, help="For padding, the max number of sentences within a document.")
    parser.add_argument("--temp", type=float, default=0.1, help="temperature scaling factor; regularization")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for training contextualized embeddings.")
    parser.add_argument("--random_state", type=int, default=42, help="random seed.")
    parser.add_argument("--iters", type=int, default=1, help="number of iters for re-training embeddings.")
    parser.add_argument("--k", type=float, default=0.075, help="Top k percent docs added to class set.")
    parser.add_argument("--doc_thresh", type=float, default=0.5, help="Pseudo-training dataset threshold.")

    args = parser.parse_args()
    print(vars(args))
    main(args)