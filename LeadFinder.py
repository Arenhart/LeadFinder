#!/usr/bin/env python
# coding: utf-8

import io

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
import shap
import ipywidgets as widgets
from ipywidgets import interactive
import streamlit as st
import numpy as np

st.title('LeadFinder')
st.image('worker.png', width=150)

st.write('Utilize o LeadFinder para encontrar clientes com perfis semelhantes ao seu portfólio atual. O aplicativo analisa uma base de dados com mais de 400.000 empresas, e mais de 100 propriedades. Além de indicar as empresas com o perfíl mais promissor, é possível avaliar para cada empresa como cada uma de suas característica contribui para a sua classificação. Também são disponibilizados as características que são mais importantes para o portfólio, e a possibilidade de comparar a distribuição de valores entre o mercado e o portfólio para cada uma das características.')

@st.cache
def get_market():
    market = pd.read_csv('estaticos_market.csv')
    market.drop('Unnamed: 0', axis=1, inplace=True)
    market.drop('Unnamed: 0.1', axis=1, inplace=True)
    market.set_index(market['id'],inplace=True)
    market.drop('id', axis=1, inplace=True)

    for col in market.columns:
        if market[col].isna().sum()/market[col].size >= 0.9:
            market.drop(col, axis=1, inplace=True)
            
    market_obj = market.select_dtypes(exclude = 'number')
    for col in market_obj.columns:
        market[col].fillna(value='NA', inplace=True)

    market_num = market.select_dtypes(include = 'number')
    inputer = SimpleImputer(strategy = 'median')
    market[market_num.columns] = inputer.fit_transform(market_num).astype(np.float32)
    return market


@st.cache
def get_prospects(market, client_list):
    prospects = market.copy()
    clients_df = pd.DataFrame({'id' : client_list})
    clients_df['is_client'] = 1
    clients_df.set_index('id', inplace=True)
    clients_df = clients_df.loc[clients_df.index.intersection(prospects.index)]
    prospects['is_client'] = 0
    prospects.loc[clients_df.index, 'is_client'] = 1
    prospects = prospects[prospects['is_client'] == 0]
    prospects.drop('is_client', axis=1, inplace=True)
    prospects = prospects[:50000]
    return prospects
    
@st.cache
def get_clients(market, client_list):
    df = pd.DataFrame({'col_1': 0}, index = client_list)
    return market.loc[df.index.intersection(market.index)]

def train_classifier(classifier, clients, prospects):
    target = pd.concat((pd.Series(1, index=clients.index),pd.Series(0, index=prospects.index)))
    cat_columns = clients.select_dtypes(exclude = 'number').columns
    classifier.fit(pd.concat((clients, prospects)), target, cat_features=cat_columns)

@st.cache    
def get_predict(classifier, prospects):
    affinity = pd.Series(classifier.predict(prospects), index = prospects.index)
    affinity /= affinity.max()
    affinity = round(affinity, 2)
    return affinity.sort_values(ascending=False)

@st.cache
def get_shap(classifier, prospects):
    explainer = shap.TreeExplainer(classifier)
    return explainer.shap_values(prospects) * 1000

@st.cache
def run_classification(file_loader):
    cat = CatBoostRegressor(silent=True, iterations=200, max_depth = 5)
    market = get_market()
    client_list = pd.read_csv(io.StringIO(file_loader.read().decode()), sep=',')['id']
    prospects = get_prospects(market, client_list)
    clients = get_clients(market, client_list)
    try:
        predict = get_predict(cat, prospects)
    except:
        train_classifier(cat, clients, prospects)
        predict = get_predict(cat, prospects)
    
    predict_shap = get_shap(cat, prospects)

    del market

    return client_list, predict, predict_shap, cat, prospects, clients


file_loader = st.file_uploader('Selecione arquivo com a lista de clientes (.csv)', type='csv', encoding=None)
if file_loader is not None:
    client_list, predict, predict_shap, cat, prospects, clients = run_classification(file_loader)
    
    st.subheader('Importância das características para o portfólio')
    affinity = cat.get_feature_importance(prettified=True)[:30]
    affinity.name = 'Afinidade relativa'
    
    cat.get_feature_importance(prettified=True)[:30]
    st.subheader('Afinidade relativa dos 10 melhores leads do mercado')
    predict[:10]

    st.subheader('Avaliação de clientes')
    st.write('Valores positivos de Efeito indicam que a caracteristica contribui para considerar a empresa como um bom lead, enquanto valores negativos indicam que essa característica diminui a afinidade')
    prospect_choice = st.selectbox(label='Escolha um cliente', options=predict.index[:30])
    if prospect_choice:
        st.write(pd.DataFrame({'Propriedade': prospects.loc[str(prospect_choice)], 'Efeito' : predict_shap[prospects.index.get_loc(prospect_choice)]}))

    st.subheader('Avaliação de características')
    feature = st.selectbox(label='Escolha de característica', options=cat.get_feature_importance(prettified=True)['Feature Id'])
    if feature:
        st.write(feature)
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.hist(prospects.loc[:, feature])
        ax2.hist(clients.loc[:, feature])
        f.suptitle(feature)
        ax1.set_title('Mercado')
        ax2.set_title('Portfolio')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        st.write(f)

