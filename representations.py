import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import argparse
import re
import pickle as pk
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from model import MEGClassModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

#### PRE-PROCESSING ####

# mainly for agnews
def clean_html(string: str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = string.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = string.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + string)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        # clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
        string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
    return string

# mainly for 20news
def clean_email(string: str):
    return " ".join([s for s in string.split() if "@" not in s])


def clean_str(string):
    string = clean_html(string)
    string = clean_email(string)
    string = re.sub(r'http\S+', '', string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\_\-\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

#### HELPER FUNCTIONS ####

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def getClassRepr(args, load=True):
    with open(os.path.join("/home/pk36/XClass/data/intermediate_data", args.dataset_name, f"document_repr_lm-bbu-12-mixture-plm.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_representations = dictionary["class_representations"]
    return class_representations

# Given tensor representations for classes and documents, identify the top class
def findMaxClass(class_repr, doc_repr, labels=None, confident=False):
    # labels (N x C): each row has either one class with a value = 1 OR all rows are zero meaning not confident enough

    class_repr = F.normalize(class_repr, dim=1) # C x emb_dim
    doc_repr = F.normalize(doc_repr, dim=1) # N x emb_dim

    # cosine similarity between doc_repr and class_repr
    cos_sim = torch.mm(doc_repr, class_repr.transpose(0,1)) # N x C

    if labels is None:
        # identify closest class i to doc
        i_sim = torch.max(cos_sim, dim=1)[0] # N x 1
    elif (not confident) and (labels is not None):
        i_sim = cos_sim[labels]
    else:
        # get the confident class cos-sim OR get the max cos-sim (1 if no confident class, 0 if yes)
        i_sim = torch.max(cos_sim * labels, dim=1)[0] + (1 - torch.sum(labels, dim=1)) * torch.max(cos_sim, dim=1)[0]
    
    return i_sim

def contrastive_loss(class_repr, doc_repr, i_sim, temp=0.2):
    class_repr = F.normalize(class_repr, dim=1) # C x emb_dim
    doc_repr = F.normalize(doc_repr, dim=1) # N x emb_dim

    # cosine similarity between doc_repr and class_repr
    cos_sim = torch.mm(doc_repr, class_repr.transpose(0,1)) # N x C

    # compute loss
    loss = -torch.log((torch.exp(i_sim)/temp)/torch.sum(torch.exp(cos_sim/temp), dim=1))

    return torch.mean(loss)

def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))

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

def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }

#### GENERATING INITIAL SENTENCE EMBEDDINGS ####

def sentenceEmb(args, data_path, new_data_path, device):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = model.to(device)
    # sent_representations = []
    padded_sent_representations = []
    sentence_mask = []

    with open(data_path, "r") as dataset_len:
        num_docs = len(dataset_len.readlines())

    with open(data_path, "r") as dataset:
        # construct the mask (N docs x S sentences; assume each has S=50 sentences and pad at end)
        trimmed = 0

        for doc in tqdm(dataset, total=num_docs):

            # Tokenize sentences
            sents = sent_tokenize(clean_str(doc))
            # Count number of sentences
            num_sent = len(sents)
            if num_sent > args.max_sent:
                # print(f'The number of sentences in this document, {num_sent}, is greater than max_sent {max_sent}')
                # Since start and ending sentences tend to be the most important, take out some middle sentences
                # end_idx = (num_sent//2) - (num_sent - max_sent)//2
                # start_idx = (num_sent//2) + (num_sent - max_sent)//2
                # sents = sents[:end_idx] + sents[start_idx:]
                trimmed += 1
                sents = sents[:args.max_sent]

            encoded_input = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(device)

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = tensor_to_numpy(F.normalize(embeddings, p=2, dim=1))

            # Add padded sentences
            if num_sent < args.max_sent:
                pad_embeddings = np.concatenate((embeddings, np.zeros((args.max_sent - num_sent, embeddings.shape[1]))), axis=0)
            else:
                pad_embeddings = embeddings

            # Update mask so that padded sentences are not included in attention computation
            curr_mask = np.array([True] * args.max_sent)
            curr_mask[:num_sent] = False

            # Add embeddings to global list
            # sent_representations.append(embeddings)
            padded_sent_representations.append(pad_embeddings)
            sentence_mask.append(curr_mask)

        # sent_representations = np.array(sent_representations)
        padded_sent_representations = np.array(padded_sent_representations)
        sentence_mask = np.array(sentence_mask)

    print(f"Trimmed Documents: {trimmed}")
    class_representations = getClassRepr(args)
    print("Retrieved Class Representations!")

    # with open(os.path.join(new_data_path, "repr.pk"), "wb") as r:
    #     repr_pickle = {
    #         "padded_sent_representations": padded_sent_representations,
    #         "sentence_mask": sentence_mask
    #     }
    #     pk.dump(repr_pickle, r, protocol=4)
    #     print("Saved initial sentence representations!")

    #return sent_representations, padded_sent_representations, sentence_mask, class_representations
    return padded_sent_representations, sentence_mask, class_representations

def contextualizedEmb(args, sent_representations, mask, class_repr, new_data_path, device):
    sent_representations = torch.from_numpy(sent_representations)
    mask = torch.from_numpy(mask)
    dataset = TensorDataset(sent_representations, mask)
    sampler = SequentialSampler(dataset)
    dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, shuffle=False)
    # sent_representations: N docs x L sentences x 1024 emb (L with padding is always max_sents=50)
    model = MEGClassModel(args.emb_dim, args.num_heads).to(device)

    total_steps = len(dataset_loader) * args.epochs / args.accum_steps
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

    print("Starting to train!")

    for i in tqdm(range(args.epochs)):
        model.train()
        total_train_loss = 0
        
        model.zero_grad()
        for j, batch in enumerate(tqdm(dataset_loader)):
            input_emb = batch[0].to(device).float()
            input_mask = batch[1].to(device)
            
            c_sent, c_doc, _ = model(input_emb, mask=input_mask)

            i_sim = findMaxClass(torch.from_numpy(class_repr).float().to(device), c_doc)

            loss = contrastive_loss(torch.from_numpy(class_repr).float().to(device), c_doc, i_sim) / args.accum_steps

            total_train_loss += loss
            loss.backward()
            optimizer.step()
            model.zero_grad()

        scheduler.step()

        avg_train_loss = torch.tensor([total_train_loss / len(dataset_loader) * args.accum_steps])
        print(f"Average training loss: {avg_train_loss.mean()}")

    model.eval()

    torch.save(model.state_dict(), os.path.join(new_data_path, f"{args.dataset_name}_model_weights.pth"))

    print("Starting to evaluate!")

    sentence_predictions = None
    doc_predictions = None

    with torch.no_grad(), open(os.path.join(new_data_path, "contextualized_sent.txt"), 'w') as fs, open(os.path.join(new_data_path, "contextualized_docs.txt"), 'w') as fd:
        for batch in tqdm(dataset_loader):
            input_emb = batch[0].to(device).float()
            input_mask = batch[1].to(device)

            c_sent, c_doc, alpha = model(input_emb, mask=input_mask)
            c_sent = tensor_to_numpy(c_sent)
            c_doc = tensor_to_numpy(c_doc)

            # for row in c_doc:
            #     fd.write(' '.join(map(str, row)) + '\n')

            # fs.write(str(c_sent))
            # fs.write("\n")

            # fd.write(str(c_doc))
            # fd.write("\n")



            sent_class = sentenceToClass(c_sent, class_repr, tensor_to_numpy(alpha))
            doc_class = docToClass(c_doc, class_repr, tensor_to_numpy(alpha))
            if sentence_predictions is None:
                sentence_predictions = sent_class
                doc_predictions = doc_class
            else:
                sentence_predictions = np.append(sentence_predictions, sent_class)
                doc_predictions = np.append(doc_predictions, doc_class)
            

            
    # return tensor_to_numpy(context_sent), tensor_to_numpy(context_doc), class_repr
    return sentence_predictions, doc_predictions


def main(args):

    data_path = os.path.join("/shared/data2/pk36/multidim/multigran", args.dataset_name, "dataset.txt")
    new_data_path = os.path.join("/home/pk36/MEGClass/intermediate_data", args.dataset_name)

    # global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # this is where we save all representations
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    padded_sent_representations, sentence_mask, class_repr = sentenceEmb(data_path, new_data_path, device, max_sent=args.max_sent)

    with open(os.path.join("/shared/data2/pk36/multidim/multigran", args.dataset_name, "labels.txt"), "r") as l:
        gold_labels = l.read().splitlines()
        gold_labels = [int(i) for i in gold_labels]


    sent_to_doc_class, doc_to_class = contextualizedEmb(args, padded_sent_representations, sentence_mask, class_repr, new_data_path, device)

    sent_pred = np.rint(sent_to_doc_class)
    doc_pred = np.rint(doc_to_class)

    print("Evaluate Predictions (Sentence-Based): ")
    evaluate_predictions(gold_labels, sent_pred)
    print("Evaluate Predictions (Document-Based): ")
    evaluate_predictions(gold_labels, doc_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="agnews")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, "
                                                            "-1 means not doing PCA.")
    parser.add_argument("--gpu", type=int, default=7, help="GPU to use; refer to nvidia-smi")
    parser.add_argument("--emb_dim", type=int, default=768, help="sentence and document embedding dimensions; all-roberta-large-v1 uses 1024.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads to use for MultiHeadAttention.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of documents.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--accum_steps", type=int, default=1, help="For training.")
    parser.add_argument("--max_sent", type=int, default=150, help="For padding, the max number of sentences within a document.")

    args = parser.parse_args()
    print(vars(args))
    main(args)