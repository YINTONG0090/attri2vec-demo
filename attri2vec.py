
# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None
    
import networkx as nx
import pandas as pd
import numpy as np
import os
import random
import re

import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#加载网络数据
filename1=r'C:\Users\殷彤\Desktop\柴山老师\id_pair.csv'
filename2=r'C:\Users\殷彤\Desktop\柴山老师\id_vector.csv'
test1=r'C:\Users\殷彤\Desktop\柴山老师\content.txt'
test2=r'C:\Users\殷彤\Desktop\柴山老师\edgeList.txt'

#导入连接集
edgelist = pd.read_csv(filename1)
edgelist.rename(columns={'id1':'source','id2':'target'},inplace=True)
edgelist["label"] = "cites"  # set the edge type


#设置名称
feature_names =['None']+["w_{}".format(ii) for ii in range(200)]+['None'] #w_200
feature_names2=["w_{}".format(ii) for ii in range(200)]
node_column_names = ["id"]+feature_names 

#导入点集，并把特征向量拆为单列
node_data = pd.read_csv(filename2)
node=pd.DataFrame(node_data)
node.index.values #index正确
split=node['vector'].str.split(',|\[|\]',expand=True) #按逗号和[]分割向量
split.columns=feature_names
split=split.drop(['None'],axis=1) #将空行删掉
node_feature1=split 
node_feature1.index.values #无index
node_all=node.join(split)
node_all=node_all.drop(['vector'],axis=1)

#bulid network
G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
print(nx.info(G_all_nx))

nx.set_node_attributes(G_all_nx,'paper','label')
#G_all_nx.nodes[53803875]['label']

node_real=pd.DataFrame(data=G_all_nx.nodes)

#nodes集中有的点未使用，将其剔除,设置为node_real
node_real.columns=['id']
node_real2=node_all[node_all['id'].isin(node_real['id'])] #nodes集中网络中的点
node_real3=node_real2.set_index(["id"],inplace=False)#需在nodes集中设置id为index

G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=node_real3)

print(G_all.info())

#构建子图
subgraph_edgelist = edgelist.loc[:765303,['source','target']] #取前80%的边构造子网络
subgraph_edgelist["label"]="cites"

G_sub_nx = nx.from_pandas_edgelist(subgraph_edgelist, edge_attr="label")

nx.set_node_attributes(G_sub_nx, "paper", "label")

subgraph_node_ids = sorted(list(G_sub_nx.nodes))

subgraph_node_features = node_real3[feature_names2].reindex(subgraph_node_ids)

G_sub = sg.StellarGraph.from_networkx(G_sub_nx, node_features=subgraph_node_features)

print(G_sub.info())

#在子图上训练attri2vec
nodes=list(G_sub.nodes())
number_of_walks=2
length=5

unsupervised_samples = UnsupervisedSampler(
    G_sub, nodes=nodes, length=length, number_of_walks=number_of_walks
)
batch_size = 50
epochs = 6
generator = Attri2VecLinkGenerator(G_sub, batch_size)
layer_sizes = [128]
attri2vec = Attri2Vec(
    layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
)

# Build the model and expose input and output sockets of attri2vec, for node pair inputs:
x_inp, x_out = attri2vec.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)


model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-2),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)

#预测节点
x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = node_real3.index
node_gen = Attri2VecNodeGenerator(G_all, batch_size).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

#获取样本内外连接
no_subgraph_edgelist = edgelist.loc[765304:,['source','target']] #取后20%的边

in_sample_edges = []
out_of_sample_edges = []





