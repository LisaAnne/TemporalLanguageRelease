# Localizing Moments in Video with Natural Language.

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Temporal Language." EMNLP (2018).

Find the paper [here](https://arxiv.org/pdf/1708.01641.pdf) and the project page [here.](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html)

```
@inproceedings{hendricks18emnlp, 
        title = {Localizing Moments in Video with Temporal Language.}, 
        author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan}, 
        booktitle = {Empirical Methods in Natural Language Processing (EMNLP)}, 
        year = {2018} 
}
```

License: BSD 2-Clause license

## Running the Code

**Preliminaries**:  I trained all my models with the [BVLC caffe version](https://github.com/BVLC/caffe).  Before you start, look at "utils/config.py" and change any paths as needed (e.g., perhaps you want to point to a Caffe build in a different folder).

**Getting Started**

Run ``setup.sh'' to download pre-extracted features, data (the TEMPO-HL and TEMPO-TL datasets), and pre-trained models.


**Training**

You can re-train all models using scripts in the ''experiments'' folder.  
To retrain the MLLC model on TEMPO-HL run X and X.
To retrain the MLLC model on TEMPO-TL run X and X.
See ``experiments/models.txt'' for a description of each experiment in the experiments folder.

**Testing**



## Data

This is a description of the data you will need to replicate my experiments.
Note that ``setup.sh'' will download everything you need to replicate my experiments and will place everything in the right folder.

### TEMPO Annotations


### Getting the Videos

Please see the instructions for downloading videos [here](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md). 


