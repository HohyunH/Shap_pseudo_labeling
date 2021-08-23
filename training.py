import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import transformers
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')

from dataset import *


def train(epochs, model, optimizer, train_loader, tokenizer, device, pl=False):
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

        torch.save(model.state_dict(), './distill_bert.pth')

def evaluate(model, eval_loader, tokenizer, device):
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

def class_same(train_df, class_num):
    zero_df = train_df[train_df['label']==0][:class_num]
    one_df = train_df[train_df['label']==1][:class_num]
    total_df = pd.concat([zero_df, one_df])
    total_df=total_df.sample(frac=1).reset_index(drop=True)

    return total_df

def main():

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

    nsmc_train_dataset = BinaryDataset(train_df)
    train_loader = DataLoader(nsmc_train_dataset, batch_size=2, shuffle=True, num_workers=2)

    nsmc_eval_dataset = BinaryDataset(val_df)
    eval_loader = DataLoader(nsmc_eval_dataset, batch_size=2, shuffle=False, num_workers=2)

    optimizer = Adam(model.parameters(), lr=1e-6)

    train(5, model, optimizer, train_loader, tokenizer, device)
    evaluate(model, eval_loader, tokenizer, device)

if __name__ == '__main__':
    main()