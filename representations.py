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
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


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

def contrastive_loss(class_repr, doc_repr, temp=0.2, labels=None, confident=False):
    # labels (N x C): each row has either one class with a value = 1 OR all rows are zero meaning not confident enough

    class_repr = F.normalize(class_repr, dim=1) # C x emb_dim
    doc_repr = F.normalize(doc_repr, dim=1) # N x emb_dim

    # cosine similarity between doc_repr and class_repr
    cos_sim = torch.mm(doc_repr, class_repr.transpose(0,1)) # N x C

    if labels is None:
        # identify closest class i to doc
        i_sim = torch.max(cos_sim, dim=1) # N x 1
    elif (not confident) and (labels is not None):
        i_sim = cos_sim[labels]
    else:
        # get the confident class cos-sim OR get the max cos-sim (1 if no confident class, 0 if yes)
        i_sim = torch.max(cos_sim * labels, dim=1) + (1 - torch.sum(labels, dim=1)) * torch.max(cos_sim, dim=1)

    # compute loss
    loss = -torch.log((torch.exp(i_sim)/temp)/torch.sum(torch.exp(cos_sim/temp), dim=1))

    return torch.mean(loss)


def sentenceEmb(data_path, new_data_path, max_sent):
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
            if num_sent > max_sent:
                # print(f'The number of sentences in this document, {num_sent}, is greater than max_sent {max_sent}')
                # Since start and ending sentences tend to be the most important, take out some middle sentences
                # end_idx = (num_sent//2) - (num_sent - max_sent)//2
                # start_idx = (num_sent//2) + (num_sent - max_sent)//2
                # sents = sents[:end_idx] + sents[start_idx:]
                trimmed += 1
                sents = sents[:max_sent]

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
            if num_sent < max_sent:
                pad_embeddings = np.concatenate((embeddings, np.zeros((max_sent - num_sent, args.emb_dim))), axis=0)
            else:
                pad_embeddings = embeddings

            # Update mask so that padded sentences are not included in attention computation
            curr_mask = np.array([True] * max_sent)
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

def contextualizedEmb(args, sent_representations, mask, class_repr, new_data_path):
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
        
        wrap_dataset_loader = tqdm(dataset_loader)
        model.zero_grad()
        for j, batch in enumerate(wrap_dataset_loader):
            input_emb = batch[0].to(device)
            input_mask = batch[1].to(device)
            
            c_sent, c_doc = model(input_emb, mask=input_mask)
            loss = contrastive_loss(torch.from_numpy(class_repr), c_doc) / args.accum_steps

            total_train_loss += loss
            loss.backward()
            optimizer.step()
            model.zero_grad()

        scheduler.step()

        avg_train_loss = torch.tensor([total_train_loss / len(dataset_loader) * args.accum_steps])
        print(f"Average training loss: {avg_train_loss.mean()}")

    train.eval()
    context_sent = None
    context_doc = None

    with torch.no_grad():
        for batch in tqdm(dataset_loader):
            input_emb = batch[0].to(device)
            input_mask = batch[1].to(device)

            c_sent, c_doc = model(input_emb, mask=input_mask)

            if context_sent is None:
                context_sent = c_sent
                context_doc = c_doc
            else:
                context_sent = torch.cat((context_sent, c_sent), dim=0)
                context_doc = torch.cat((context_doc, c_doc), dim=0)


    # with open(os.path.join(new_data_path, "context_repr.pk"), "wb") as r:
    #     repr_pickle = {
    #         "contextualized_sent":tensor_to_numpy(context_sent),
    #         "contextualized_doc":tensor_to_numpy(context_doc),
    #         "class_representations": class_repr
    #     }
    #     pk.dump(repr_pickle, r, protocol=4)

    torch.save(model.state_dict(), os.path.join(new_data_path, f"{args.dataset_name}_model_weights.pth"))

    return tensor_to_numpy(context_sent), tensor_to_numpy(context_doc), class_repr


def main(args):

    data_path = os.path.join("/shared/data2/pk36/multidim/multigran", args.dataset_name, "dataset.txt")
    new_data_path = os.path.join("/home/pk36/MEGClass/intermediate_data", args.dataset_name)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # this is where we save all representations
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    padded_sent_representations, sentence_mask, class_repr = sentenceEmb(data_path, new_data_path, max_sent=args.max_sent)
    csent, cdoc, class_repr = contextualizedEmb(args, padded_sent_representations, sentence_mask, class_repr, new_data_path)


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