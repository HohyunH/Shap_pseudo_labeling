import shap
import torch
import scipy as sp
import numpy as np
import pandas as pd

import transformers

from dataset import *

def make_shap_data(df, number):

  shap_dict={}
  data_df = df.reset_index()
  review = data_df['review']
  label = data_df['label']
  shap_dict['label'] = label[:number].tolist()
  shap_dict['text'] = review[:number].tolist()

  return shap_dict

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

    for test_step in range(int(len(test_df)/1000)):
        tests_df = test_split(test_step, test_df)

        nsmc_test_dataset = BinaryDataset(tests_df)
        test_loader = DataLoader(nsmc_test_dataset, batch_size=2, shuffle=False, num_workers=2)

        explainer = shap.Explainer(f, tokenizer)
        shap_values = explainer(make_shap_data(tests_df, 30), fixed_context=1)

