import shap
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import transformers

from dataset import *

import torch
from torch.optim import Adam


def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    # The logit function is defined as logit(p) = log(p/(1-p)). Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p<0 or p>1 yields nan.
    return val


def class_same(train_df, class_num):
    zero_df = train_df[train_df['label'] == 0][:class_num]
    one_df = train_df[train_df['label'] == 1][:class_num]
    total_df = pd.concat([zero_df, one_df])
    total_df = total_df.sample(frac=1).reset_index(drop=True)

    return total_df

def test_split(id, test_df):
    split_df = test_df[id*1000:(id+1)*1000]
    return split_df

def check_shap(df, index_list:list):

    shap_dict={}
    data_df = df.reset_index()
    review = data_df['review']
    label = data_df['label']
    shap_dict['label'] = [label.tolist()[i] for i in index_list]
    shap_dict['text'] = [review.tolist()[i] for i in index_list]

    return shap_dict

def evaluate(model, eval_loader, tokenizer):
    model.eval()

    total_len = 0
    total_correct = 0
    with torch.no_grad():
        for text, label in eval_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e[:512] + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            _, logits = outputs[0], outputs[1]

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)

    print('Test accuracy: ', total_correct / total_len)

def in_testing(model, eval_loader, tokenizer, device, thres : float):
    model.eval()

    total_len = 0
    total_correct = 0
    over_thres = []
    under_thres = []
    over_label = []
    idx = 0
    with torch.no_grad():
        for text, label in tqdm_notebook(eval_loader):
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e[:512] + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            _, logits = outputs[0], outputs[1]

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            bool_list =  torch.max(F.softmax(logits), dim=1)[0].ge(thres)
            # print(bool_list)
            over_thres.extend([i+idx for i, tr in enumerate(bool_list) if tr == True])
            over_label.extend([pred[i] for i, tr in enumerate(bool_list) if tr == True])
            under_thres.extend([j+idx for j,fa in enumerate(bool_list) if fa == False])
            idx+=len(labels)

    print('prediction accuracy: ', total_correct / total_len)
    return over_thres, under_thres, over_label

def shap_rule(num, shap_values):
    sorted_shap = sorted(shap_values[num].values)
    sorted_shap_rev = sorted(shap_values[num].values, reverse=True)
    score = np.mean(sorted_shap_rev[:5]) - np.mean(sorted_shap[:5])
    return score

def idx_list(idx, shap_values):
  index_list = []

  numbers = [j for j in range(len(shap_values[idx].data))]
  sentences = sent_tokenize("".join(shap_values[idx].data))

  for i in range(len(sentences)):
    tmp = []
    tkns = [tokenizer.encode(sentences[i], padding='max_length', max_length=512, truncation=True)]
    tkns = [k for k in tkns[0] if k != 0]
    if i == 0:
      tkns.pop(-1)
    elif i == len(sentences)-1:
      tkns.pop(0)
    else :
      tkns.pop(0)
      tkns.pop(-1)
    for t in tkns:
      tmp.append(numbers[0])
      numbers.pop(0)
    index_list.append(tmp)

  return index_list, sentences

def pos_neg_sentence(idx, shap_values):
  idx_lists, sentences = idx_list(idx, shap_values)

  tmp = []
  for s in range(len(sentences)):
    idx_ = idx_lists[s]
    t = 0
    for i in idx_:
      t+=shap_values[idx].values[i]
    # tmp.append(t/len(idx_))
    tmp.append(t)

  max_ = np.argmax(tmp)
  min_ = np.argmin(tmp)

  return sentences[max_], sentences[min_]

def extract_data(tests_df):

    pos_sentences = []
    neg_sentences = []
    nsmc_test_dataset = BinaryDataset(tests_df)
    test_loader = DataLoader(nsmc_test_dataset, batch_size=2, shuffle=False, num_workers=2)

    p_over, p_under, p_label = in_testing(model, test_loader, tokenizer, device, 0.95)

    print(f"original pseudo_labeling data : {len(p_over)}")
    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(check_shap(tests_df, p_under), fixed_context=1)

    print("****************** splitting pos&neg *******************")
    for i in tqdm(range(len(p_under))):
        if shap_rule(i) > 2:
            pos, neg = pos_neg_sentence(i, shap_values)
            pos_sentences.append(pos); neg_sentences.append(neg)

    pos_df = pd.DataFrame({"review": pos_sentences, "label": 0})
    neg_df = pd.DataFrame({"review": neg_sentences, "label": 1})
    under_df = pd.concat([pos_df, neg_df]).reset_index().drop(['index'], axis='columns')

    plus_dataset = BinaryDataset(tests_df)
    plus_loader = DataLoader(plus_dataset, batch_size=2, shuffle=False, num_workers=2)

    p_sh_over, _, p_sh_label = in_testing(model, plus_loader, tokenizer, device, 0.95)
    print(f"with explainable pseudo_labeling data : {len(p_sh_label)}")

    original_df = pd.DataFrame({"review": [tests_df['review'].tolist()[i] for i in p_over], "label": p_label})
    plus_df = pd.DataFrame({"review": [under_df['review'].tolist()[i] for i in p_sh_over], "label": p_sh_label})

    extract_df = pd.concat([original_df, plus_df])

    return extract_df, p_under

def train(epochs, model, optimizer, train_loader, tokenizer, device, path, pl=False):
    itr = 1
    p_itr = 100
    total_loss = 0
    total_len = 0
    total_correct = 0

    model.train()
    for epoch in range(epochs):

        for text, label in train_loader:
            optimizer.zero_grad()
            if pl == False:
                encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]

                padded_list = [e[:512] + [0] * (512 - len(e)) for e in encoded_list]

                sample = torch.tensor(padded_list)
            else:
                sample = text
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs[0], outputs[1]

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if itr % p_itr == 0:
                print(
                    '[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, epochs, itr,
                                                                                                total_loss / p_itr,
                                                                                                total_correct / total_len))
                total_loss = 0
                total_len = 0
                total_correct = 0

            itr += 1

        torch.save(model.state_dict(), './distill_bert_{}.pth'.format(path))

if __name__=="__main__":

    yelp = pd.read_csv('./yelp.csv')
    yelp['label'] = yelp['rating'] - 1
    del yelp['rating']

    train_df = yelp[:1000]
    test_df = yelp[1000:-1000]
    val_df = yelp[-1000:]

    train_df = class_same(train_df, 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available devices ', torch.cuda.device_count())

    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = model.to(device)
    model.load_state_dict(torch.load('./distill_bert.pth'))
    optimizer = Adam(model.parameters(), lr=1e-6)
    epochs = 5

    for test_step in range(int(len(test_df)/1000)):

        tests_df = test_split(test_step, test_df)

        extract_df, p_under = extract_data(tests_df)

        train(epochs, model, optimizer, train_loader, tokenizer, device, str(test_step), pl=False)

        model.zero_grad()




