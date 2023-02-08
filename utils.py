import numpy as np
import os
import pickle as pk
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, f1_score

DATA_FOLDER_PATH = os.path.join("/shared/data2/pk36/multidim/multigran")
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join("/home/pk36/MEGClass/intermediate_data")


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

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
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_micro": f1_micro,
            "f1_macro": f1_macro
        }
    
def getSentClassRepr(args):
    with open(os.path.join("/home/pk36/XClass/data/intermediate_data", args.dataset_name, f"document_repr_lm-bbu-12-mixture-plm.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_repr = dictionary["class_representations"]
        sent_repr = dictionary["sent_representations"]
    return sent_repr, class_repr


def getDSMapAndGold(args, sent_dict):
    # get the ground truth labels for all documents and assign a "ground truth" label to each sentence based on its parent document
    gold_labels = list(map(int, open(os.path.join("/shared/data2/pk36/multidim/multigran", args.dataset_name, "labels.txt"), "r").read().splitlines()))
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
            # w = np.ones(doc_lengths[doc_id])/doc_lengths[doc_id]
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