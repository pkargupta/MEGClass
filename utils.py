import numpy as np
import os
import pickle as pk
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, f1_score
import torch
from shutil import copyfile
from transformers import BertModel, BertTokenizer

DATA_FOLDER_PATH = "/shared/data2/pk36/multidim/multigran/"
INTERMEDIATE_DATA_FOLDER_PATH = "/home/pk36/MEGClass/intermediate_data/"

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased')
}


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sentenceToClass(sent_repr, class_repr, weights):
    # sent_repr: N x S x E
    # class_repr: C x E
    # weights: N x S # equals 0 for masked sentences

    #cos-sim between (N x S) x E and (C x E) = N x S x C
    m, n = sent_repr.shape[:2]
    sentcos = cosine_similarity(sent_repr.reshape(m*n,-1), class_repr).reshape(m,n,-1)
    sent_to_class = np.argmax(sentcos, axis=2) # N x S
    sent_to_doc_class = np.sum(np.multiply(sent_to_class, weights), axis=1) # N x 1
    return sent_to_doc_class

def docToClass(doc_repr, class_repr):
    # doc_repr: N x E
    # class_repr: C x E

    #cos-sim between N x E and C x E = N x C
    doccos = cosine_similarity(doc_repr, class_repr)
    doc_to_class = np.argmax(doccos, axis=1) # N x 1
    return doc_to_class

def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False, return_confusion=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if return_confusion and output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    if output_to_console:
        print("F1 micro: " + str(f1_micro))
        print("F1 macro: " + str(f1_macro))
    if return_tuple:
        return f1_macro, f1_micro
    else:
        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro
        }
    
def getSentClassRepr(args):
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name, f"document_repr_lm-bbu-12-mixture-plm.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_repr = dictionary["class_representations"]
        sent_repr = dictionary["sent_representations"]
        if (dictionary.has_key("document_representations")):
            doc_repr = dictionary["document_representations"]
            return doc_repr, sent_repr, class_repr
        else:
            return sent_repr, class_repr


def getDSMapAndGold(args, sent_dict):
    # get the ground truth labels for all documents and assign a "ground truth" label to each sentence based on its parent document
    gold_labels = list(map(int, open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, "labels.txt"), "r").read().splitlines()))
    gold_sent_labels = []
    # get all sent ids for each doc
    doc_to_sent = []
    sent_id = 0
    for doc_id, doc in enumerate(sent_dict.values()):
        sent_ids = []
        for sent in doc:
            sent_ids.append(sent_id)
            gold_sent_labels.append(gold_labels[doc_id])
            sent_id += 1
        doc_to_sent.append(sent_ids)
            
    return gold_labels, gold_sent_labels, doc_to_sent

# get target class distribution for each document based on sentences and initial word-based class representations
def getTargetClasses(padded_sent_repr, doc_lengths, class_repr, weights=None):
    # weights: N x 150
    class_weights = np.zeros((padded_sent_repr.shape[0], class_repr.shape[0])) # N x C
    sent_weights = np.zeros(padded_sent_repr.shape[:2])

    for doc_id in tqdm(np.arange(padded_sent_repr.shape[0])):
        l = doc_lengths[doc_id]
        sent_emb = padded_sent_repr[doc_id, :l, :] # S x E
        sentcos = cosine_similarity(sent_emb, class_repr) # S x C
        sent_to_class = np.argmax(sentcos, axis=1) # S

        # default: equal vote weight between all sentences
        if weights is None:
            # top cos-sim - second cos-sim
            toptwo = np.partition(sentcos, -2)[:, -2:] # S x 2
            toptwo = toptwo[:, 1] - toptwo[:, 0] # S
            w = toptwo / np.sum(toptwo)
            sent_weights[doc_id, :l] = w
        else:
            w = weights[doc_id, :l]
        
        class_weights[doc_id, :] = np.bincount(sent_to_class, weights=w, minlength=class_repr.shape[0])

    if weights is None:
        return class_weights, sent_weights
    else:
        return class_weights

# get target class distribution for each document based on sentences and class set
def getTargetClassSet(padded_sent_repr, doc_lengths, class_set, alpha=None, set_weights=None):
    # sent_weights: N x 150 -> weigh each sentence based on its contribution to the document
    # set_weights: C x CD -> how confident each class-indicative document is

    class_weights = np.zeros((padded_sent_repr.shape[0], len(class_set))) # N x C
    if alpha is None:
        sent_weights = np.zeros(padded_sent_repr.shape[:2]) # N x 150

    for doc_id in tqdm(np.arange(padded_sent_repr.shape[0])):
        l = doc_lengths[doc_id]
        sent_emb = padded_sent_repr[doc_id, :l, :] # S x E
        sentclass_dist = np.zeros((l, len(class_set))) # S x C

        for class_id in np.arange(len(class_set)):
            sentcos = cosine_similarity(sent_emb, class_set[class_id]) # S x CD
            if set_weights is None:
                sentclass_sim = np.mean(sentcos, axis=1) # on average, how similar each sentence is to the class set
            else:
                sentclass_sim = np.average(sentcos, axis=1, weights=set_weights[class_id]) # same but weighted average based on class-indicativeness
            
            sentclass_dist[:, class_id] = sentclass_sim


        sent_to_class = np.argmax(sentclass_dist, axis=1) # S
        
        # default: equal vote weight between all sentences
        if alpha is None:
            # top cos-sim - second cos-sim
            toptwo = np.partition(sentclass_dist, -2)[:, -2:] # S x 2
            toptwo = toptwo[:, 1] - toptwo[:, 0] # S
            w = toptwo / np.sum(toptwo)
            sent_weights[doc_id, :l] = w
        else:
            w = alpha[doc_id, :l]
        
        class_weights[doc_id, :] = np.bincount(sent_to_class, weights=w, minlength=len(class_set))

    if alpha is None:
        return class_weights, sent_weights
    else:
        return class_weights

def docToClassSet(doc_emb, class_set, set_weights=None):
    # set_weights: C x CD -> how confident each class-indicative document is
    class_dist = np.zeros((doc_emb.shape[0], len(class_set))) # D x C
    for class_id in np.arange(len(class_set)):
        doccos = cosine_similarity(doc_emb, class_set[class_id]) # D x CD
        if set_weights is None:
            cos_sim = np.mean(doccos, axis=1) # on average, how similar each document is to the class set
        else:
            cos_sim = np.average(doccos, axis=1, weights=set_weights[class_id]) # same but weighted average based on class-indicativeness
        class_dist[:, class_id] = cos_sim

    return np.argmax(class_dist, axis=1), class_dist

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
            f.write(",".join(map(str, line)))
            f.write("\n")

    copyfile(os.path.join(data_path, "classes.txt"),
             os.path.join(new_data_path, "classes.txt"))
    
def generateDataset(documents_to_class, ranks, class_dist, num_classes, cleaned_text, gold_labels, data_path, new_data_path, thresh=0.5, write=False):
    pseudo_document_class_with_confidence = [[] for _ in range(num_classes)]
    for i in range(documents_to_class.shape[0]):
        pseudo_document_class_with_confidence[documents_to_class[i]].append((ranks[i], i))

    selected = []
    confident_documents = [[] for _ in range(num_classes)]

    for i in range(num_classes):
        pseudo_document_class_with_confidence[i] = sorted(pseudo_document_class_with_confidence[i], key=lambda x: x[0], reverse=True)
        num_docs_to_take = int(len(pseudo_document_class_with_confidence[i]) * thresh)
        confident_documents[i] = pseudo_document_class_with_confidence[i][:num_docs_to_take]
        selected.extend([x[1] for x in confident_documents[i]])
    
    selected = sorted(selected)
    text = [cleaned_text[i] for i in selected]
    classes = [documents_to_class[i] for i in selected]
    probs = [class_dist[i] for i in selected]
    ###
    gold_classes = [gold_labels[i] for i in selected]
    evaluate_predictions(gold_classes, classes)
    ###
    if write:
        write_to_dir(text, classes, probs, data_path, new_data_path)
    return confident_documents

def updateClassSet(confident_docs, doc_emb, class_set):
    for i in np.arange(len(class_set)):
        doc_ind = [d[1] for d in confident_docs[i]]
        class_set[i] = np.concatenate((np.expand_dims(class_set[i][0], axis=0), doc_emb[doc_ind, :]), axis=0)
    return

def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))

def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)