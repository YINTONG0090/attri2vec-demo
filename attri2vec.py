
##### verify that we're using the correct version of StellarGraph for this notebook
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

#####input data
filename1=r'C:\Users\殷彤\Desktop\柴山老师\id_pair.csv'
filename2=r'C:\Users\殷彤\Desktop\柴山老师\id_vector.csv'

node_data = pd.read_csv(filename2)
node=pd.DataFrame(node_data)
edgelist = pd.read_csv(filename1)
edgelist.rename(columns={'id1':'source','id2':'target'},inplace=True)
edgelist["label"] = "cites"  # set the edge type

#Set the feature name
feature_names =['None']+["w_{}".format(ii) for ii in range(200)]+['None'] #w_200
feature_names2=["w_{}".format(ii) for ii in range(200)]
node_column_names = ["id"]+feature_names 

#Split the vector into columns
split=node['vector'].str.split(',|\[|\]',expand=True) #split
split.columns=feature_names
split=split.drop(['None'],axis=1) #delet ' '
node_feature1=split 
node_all=node.join(split)
node_all=node_all.drop(['vector'],axis=1)

#####bulid network by networkX
G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
print(nx.info(G_all_nx))

nx.set_node_attributes(G_all_nx,'paper','label')

#select the nodes needed to build the network in node_real set
node_real=pd.DataFrame(data=G_all_nx.nodes)
node_real.columns=['id']
node_real2=node_all[node_all['id'].isin(node_real['id'])] 
node_real3=node_real2.set_index(["id"],inplace=False)#Set the 'id' to Index(important)

#Convert the network built by NetworkX into a Stellargraph form
G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=node_real3)
print(G_all.info())

#####Build a subnetwork
subgraph_edgelist = edgelist.loc[:765303,['source','target']] #Use the first 80% of links to build the subnetwork
subgraph_edgelist["label"]="cites"

G_sub_nx = nx.from_pandas_edgelist(subgraph_edgelist, edge_attr="label")

nx.set_node_attributes(G_sub_nx, "paper", "label")

subgraph_node_ids = sorted(list(G_sub_nx.nodes))

subgraph_node_features = node_real3[feature_names2].reindex(subgraph_node_ids)

G_sub = sg.StellarGraph.from_networkx(G_sub_nx, node_features=subgraph_node_features)

print(G_sub.info())

#####Train the attri2vec model in the subnetwork  
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

#####prediction





