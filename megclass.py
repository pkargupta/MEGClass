import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import sys
import torch.nn.functional as F
import argparse
import re
import pickle as pk
from tqdm import tqdm
from scipy.special import softmax
from sklearn.decomposition import PCA
from shutil import copyfile
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, f1_score
from utils import tensor_to_numpy, docToClass, getTargetClasses, getSentClassRepr, evaluate_predictions, getDSMapAndGold
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

class MEGClassModel(nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.0):
        super(MEGClassModel, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(D_in)
        # self.embd = nn.Sequential(
        #     nn.Linear(D_in, 2*D_in),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2*D_in, D_in))
        self.embd = nn.Linear(D_in,D_hidden)
        self.attention = nn.Linear(D_hidden,1)
        
    def forward(self, x_org, mask=None):
        x, mha_w = self.mha(x_org,x_org,x_org,key_padding_mask=mask)
        x = self.layernorm(x_org+x)
        
        x = self.embd(x)
        x = torch.tanh(x) # contextualized sentences
        a = self.attention(x)
        if mask is not None:
            a = a.masked_fill_((mask == 1).unsqueeze(-1), float('-inf'))
        w = torch.softmax(a, dim=1) # alpha_k
        o = torch.matmul(w.permute(0,2,1), x) #doc 
        return o, mha_w, w, x # contextualized doc, multi-head attention weights, alpha_k, contextualized sent

def weighted_contrastive_loss(args, sample_outputs, class_weights, class_embds):
    # k: B x C, class_weights: B x C
    k = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[:,None], class_embds, axis=2)/args.temp)
    weighted_loss = -1 * (torch.log(k/(k.sum(dim=1).unsqueeze(-1))) * class_weights).sum() # B x C -> B
    return weighted_loss/len(sample_outputs)


def bertSentenceEmb(args, doc_to_sent, sent_repr):
    num_docs = len(doc_to_sent)
    doc_lengths = np.zeros(num_docs, dtype=int)
    # init_doc_repr = np.zeros((num_docs, args.emb_dim))
    padded_sent_repr = np.zeros((num_docs, args.max_sent, args.emb_dim))
    sentence_mask = np.ones((num_docs, args.max_sent))
    trimmed = 0


    for doc_id in tqdm(np.arange(num_docs)):
        start_sent = doc_to_sent[doc_id][0]
        end_sent = doc_to_sent[doc_id][-1]
        num_sent = end_sent - start_sent + 1
        if num_sent > args.max_sent:
            end_sent = start_sent + args.max_sent - 1
            num_sent = args.max_sent
            trimmed += 1
        embeddings = sent_repr[start_sent:end_sent+1]

        # save the number of sentences in each document
        doc_lengths[doc_id] = int(num_sent)

        # Add initial doc representation
        # init_doc_repr[doc_id, :] = np.mean(embeddings, axis=0)
        # Add padded sentences
        padded_sent_repr[doc_id, :embeddings.shape[0], :] = embeddings
        # Update mask so that padded sentences are not included in attention computation
        sentence_mask[doc_id, :num_sent] = 0

    
    print(f"Trimmed Documents: {trimmed}")

    return padded_sent_repr, doc_lengths, sentence_mask

def contextEmb(args, sent_representations, mask, class_repr, class_weights, 
                doc_lengths, new_data_path, device):
    sent_representations = torch.from_numpy(sent_representations)
    mask = torch.from_numpy(mask).to(torch.bool)
    class_weights = torch.from_numpy(class_weights)
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

            loss = weighted_contrastive_loss(args, c_doc, input_weights, torch.from_numpy(class_repr).float().to(device)) / args.accum_steps

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
    attention_weights = np.zeros_like(mask)
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
    
    updated_class_weights = getTargetClasses(updated_sent_repr, doc_lengths, class_repr, attention_weights)

    return doc_predictions, final_doc_emb, updated_sent_repr, attention_weights, updated_class_weights

def write_to_dir(text, labels, prob, data_path, new_data_path):
    assert len(text) == len(labels)
    print("Saving files in:", new_data_path)
    
    with open(os.path.join(new_data_path, "dataset.txt"), "w") as f:
        for i, line in enumerate(text):
            f.write(line)
            f.write("\n")

    with open(os.path.join(new_data_path, "labels.txt"), "w") as f:
        for i, line in enumerate(labels):
            f.write(str(line))
            f.write("\n")

    with open(os.path.join(new_data_path, "probs.txt"), "w") as f:
        for i, line in enumerate(prob):
            f.write(str(line))
            #f.write(",".join(map(str, line)))
            f.write("\n")

    copyfile(os.path.join(data_path, "classes.txt"),
             os.path.join(new_data_path, "classes.txt"))


def generateDataset(documents_to_class, prob, num_classes, cleaned_text, gold_labels, data_path, new_data_path):
    pseudo_document_class_with_confidence = [[] for _ in range(num_classes)]
    for i in range(documents_to_class.shape[0]):
        pseudo_document_class_with_confidence[documents_to_class[i]].append((prob[i], i))

    selected = []
    confidence_threshold = 0.5
    for i in range(num_classes):
        pseudo_document_class_with_confidence[i] = sorted(pseudo_document_class_with_confidence[i], key=lambda x: x[0], reverse=True)
        num_docs_to_take = int(len(pseudo_document_class_with_confidence[i]) * confidence_threshold)
        confident_documents = pseudo_document_class_with_confidence[i][:num_docs_to_take]
        confident_documents = [x[1] for x in confident_documents]
        selected.extend(confident_documents)
    
    selected = sorted(selected)
    text = [cleaned_text[i] for i in selected]
    classes = [documents_to_class[i] for i in selected]
    probs = [prob[i] for i in selected]
    ###
    gold_classes = [gold_labels[i] for i in selected]
    evaluate_predictions(gold_classes, classes)
    ###
    write_to_dir(text, classes, probs, data_path, new_data_path)



def main(args):
    data_path = os.path.join("/shared/data2/pk36/multidim/multigran", args.dataset_name)
    new_data_path = os.path.join("/home/pk36/MEGClass/intermediate_data", args.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # this is where we save all representations
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # read in sentence + class representations and labels
    sent_repr, class_repr = getSentClassRepr(args)
    num_classes = class_repr.shape[0]
    with open(os.path.join("/home/pk36/XClass/data/intermediate_data", args.dataset_name, "dataset.pk"), "rb") as f:
        dataset = pk.load(f)
        sent_dict = dataset["sent_data"]
        cleaned_text = dataset["cleaned_text"]
    
    gold_labels, gold_sent_labels, doc_to_sent = getDSMapAndGold(args, sent_dict)

    # BERT-Based Sentence Embeddings, Initial Doc Embeddings, & Class Representations
    # Get Class Weights
    padded_sent_repr, doc_lengths, sentence_mask = bertSentenceEmb(args, doc_to_sent, sent_repr)
    init_class_weights, init_sent_weights = getTargetClasses(padded_sent_repr, doc_lengths, class_repr, None)
    evaluate_predictions(gold_labels, np.argmax(init_class_weights, axis=1))

    curr_sent_repr = padded_sent_repr
    curr_class_weights = init_class_weights
    for i in np.arange(args.iters):
        print(f"Iter {i}: Training!")
        # train contextualized embeddings
        doc_to_class, final_doc_emb, updated_sent_repr, updated_sent_weights, updated_class_weights = contextEmb(args, curr_sent_repr, sentence_mask, class_repr, curr_class_weights, doc_lengths, new_data_path, device)
        doc_pred = np.rint(doc_to_class)
        print("Evaluate Predictions (Document-Based): ")
        evaluate_predictions(gold_labels, doc_pred)
        curr_sent_repr = updated_sent_repr
        # curr_class_weights = updated_class_weights


    # run PCA

    _pca = PCA(n_components=args.pca, random_state=args.random_state)
    pca_doc_repr = _pca.fit_transform(final_doc_emb)
    pca_class_repr = _pca.transform(class_repr)
    print(f"Explained document variance: {sum(_pca.explained_variance_ratio_)}")
    cosine_similarities = cosine_similarity(pca_doc_repr, pca_class_repr)
    doc_class_assignment = np.argmax(cosine_similarities, axis=1)
    doc_class_probs = cosine_similarities[np.arange(pca_doc_repr.shape[0]), doc_class_assignment]

    print("Evaluate Document Cosine Similarity Predictions: ")
    evaluate_predictions(gold_labels, doc_class_assignment)

    # get cleaned text
    cleaned_text = dataset["cleaned_text"]

    # generate pseudo training dataset
    generateDataset(doc_class_assignment, doc_class_probs, num_classes, cleaned_text, gold_labels, data_path, new_data_path)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="agnews")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, -1 means not doing PCA.")
    parser.add_argument("--gpu", type=int, default=7, help="GPU to use; refer to nvidia-smi")
    parser.add_argument("--emb_dim", type=int, default=768, help="sentence and document embedding dimensions; all-roberta-large-v1 uses 1024.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads to use for MultiHeadAttention.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of documents.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--accum_steps", type=int, default=1, help="For training.")
    parser.add_argument("--max_sent", type=int, default=150, help="For padding, the max number of sentences within a document.")
    parser.add_argument("--temp", type=float, default=0.2, help="temperature scaling factor; regularization")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for training contextualized embeddings.")
    parser.add_argument("--random_state", type=int, default=42, help="random seed.")
    parser.add_argument("--iters", type=int, default=1, help="number of iters for re-training embeddings.")
    args = parser.parse_args()
    print(vars(args))
    main(args)