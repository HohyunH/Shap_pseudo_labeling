import shap
import scipy as sp
import transformers

from dataset import *

import torch
from torch.optim import Adam


def check_shap(df, index_list:list):

    shap_dict={}
    data_df = df.reset_index()
    review = data_df['review']
    label = data_df['label']
    shap_dict['label'] = [label.tolist()[i] for i in index_list]
    shap_dict['text'] = [review.tolist()[i] for i in index_list]

    return shap_dict

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

    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(make_shap_data(val_df, 10), fixed_context=1)

    print(shap_values)
