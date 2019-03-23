#!/bin/bash

mkdir -p data
mkdir -p prototxts
mkdir -p prototxts/tempoTL
mkdir -p prototxts/tempoHL
mkdir -p prototxts/didemo
mkdir -p snapshots
mkdir -p cache_results
mkdir -p cache_results/tempoTL
mkdir -p cache_results/tempoHL
mkdir -p cache_results/didemo

cd data

#Download pre-processed TempoHL, TempoTL, and DiDeMo 
wget https://people.eecs.berkeley.edu/~lisa_anne/tempo/initial_release_data.zip
unzip initial_release_data.zip
mv initial_release_data/* .
rm -r initial_release_data.zip
rm -r initial_release_data

#Download prereleased features
wget https://people.eecs.berkeley.edu/~lisa_anne/tempo/average_rgb_feats.h5
wget https://people.eecs.berkeley.edu/~lisa_anne/tempo/average_flow_feats.h5

#Download glove embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.50d.txt
rm glove.6B.100d.txt
rm glove.6B.200d.txt
rm glove.6B.zip

#Download pre-released models:
cd ..
wget https://people.eecs.berkeley.edu/~lisa_anne/tempo/release_models.zip
unzip release_models.zip
rm release_models.zip 

