# Localizing Moments in Video with Temporal Language

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Temporal Language." EMNLP (2018).

Find the paper [here](https://arxiv.org/pdf/1809.01337.pdf) and the project page [here.](https://people.eecs.berkeley.edu/~lisa_anne/tempo.html)

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

**Preliminaries**

I trained all my models with the [BVLC caffe version](https://github.com/BVLC/caffe).  Please make sure you have this on your python path.

**Getting Started**

Run [setup.sh](setup.sh) to download pre-extracted features, data (TEMPO-HL and TEMPO-TL), and pre-trained models.  If you are only interested in the data (not using/replicating my code), please download the datasets [here](https://people.eecs.berkeley.edu/~lisa_anne/tempo/initial_release_data.zip).

### Training and Evaluating Models

**Replicating Results**

The bash scripts ```table3.sh```, ```table4.sh```, and ```table5.sh``` should replicate the results corresponding to those tables in the main paper.  The scripts call on ```experiments/eval_released.sh``` which is documented below.  The numbers should be very close to those reported in the paper, but there are some small differences due to bugs I caught while cleaning up the code.  See logs in ```outfiles``` folder if you would like to double check that your numbers match mine.

**Evaluating Released Models**

To evaluate released models, look at ```experiments/eval_released.sh```.  This bash script takes the following inputs:

* Model type (M): e.g., 'mllc' or 'mcn'
* Dataset (D): 'tempoHL', 'tempoTL', or 'didemo'
* GPU (G): which GPU to evaluate on
* Quick (Q): when the models are evaluated, intermediate results are cached.  After evaluationg once, you can add the flag -Q for quicker evaluation for that model in the future.

```
./experiments/eval_released.sh -M mllc -D Tempo-HL -G 0
```

Run ```./experiments/eval_released.sh -H``` for help.


**Training Your Own Models**

You train and evaluate models using scripts in the ```experiments``` folder.  To train models, look at the ```experiments/train.sh```.  This takes the following inputs:

* Model type (M): e.g., 'mllc' or 'mcn'
* Dataset (D): 'tempoHL', 'tempoTL', or 'didemo'
* Features (F): 'rgb' or 'flow' -- best results are achieved by training both and fusing outputs at test time
* Snapshot folder (S): Where you would like to save your snaphots.  Snapshots automatically saved in ```snapshots``` folder
* GPU (G): which GPU to train on

Example commands are:

```
./experiments/train.sh -M mllc -D tempoHL -F rgb -G 0
./experiments/train.sh -M mllc -D tempoHL -F flow -G 1
```
The first command trains mllc on tempoHL with rgb features on GPU 0 and the second command trains mllc on tempoHL with flow features on GPU 1.  Snapshots will automatically be saved into the ```snapshots``` folder with tags ```rgb_mllc_tempoHL``` and ```flow_mllc_tempoHL```

Run ```./experiments/train.sh -H``` for help.


**Evaluating Your Own Models**

To evaluate models, look at the ```experiments/eval.sh```.  This bash script takes the following inputs:

* Model type (M): e.g., 'mllc' or 'mcn'
* Dataset (D): 'tempoHL' or 'tempoTL'
* RGB model (R): tag for your trained RGB model
* Flow model (F): tag for your trained flow model
* GPU (G): which GPU to evaluate on
* Quick (Q): when the models are evaluated, intermediate results are cached.  After evaluationg once, you can add the flag -Q for quicker evaluation for that model in the future.

An example command is:

```
./experiments/eval.sh -R rgb_mllc_tempoHL -F flow_mllc_tempoHL -D tempoHL -M mllc
```

This will evaluate the two models, corresponding to snapshots saved in ```flow_mllc_tempoHL``` and ```rgb_mllc_tempoHL```, trained using the train commands above.
After you have evaluated the models once, use the flag ```-Q``` to used cached values and speed up the evaluation.

Run ```./experiments/eval.sh -H``` for help.

You will probably see slightly different numbers if you re-train and evaluate the models.  This is because the models are sensitive to the python random seed.  I trained five models with different random seeds and report the mean R@1, R@5, and mIoU (average across sentence types below) on Tempo-HL below:

| | Rank@1 | Rank@5 | mIOU |
| --- | --- | --- | --- |
| MCN | 19.50 | 70.73 | 41.92 |
| MLLC-global | 19.64 | 71.21 | 42.48 |
| MLLC-ba | 20.54 | 70.95 | 42.90 |
| MLLC | 20.71 | 71.72 | 44.20 |

## Data

This is a description of the data you will need to replicate my experiments.
Note that ``setup.sh``` will download everything you need to replicate my experiments and will place everything in the appropriate folder.

### TEMPO Annotations

The TEMPO annotations will have the following fields:

* annotation_id: Each annotation_id is asigned as {temporal_word}\_{didemo_id}\_{N} where the temporal_word is "before", "after", "then" or (in TEMPO-HL) "while".  The didemo_id is the didemo id for the context moment. In TEMPO-TL, there are multiple temporal sentences for each DiDeMo context moement which we identify with N.
* context: The ground truth context for the sentence.
* description
* reference_description: Reference description given to the AMT workers.  Note that we did not provide reference descriptions when collecting "while" annotations.  For "then" annotations in TEMPO-TL there will be two reference descriptions.
* times: Three ground truth human annotations.  All three are used in evaluation.  See the paper for details.
* video

Note that I train my models on both TEMPO and DiDeMo annotations.  Both DiDeMo and TEMPO have multiple annotated temporal segments per video.  In DiDeMo, sometimes one of the annotations is an outlier (e.g., three annotations correspond to seconds 10-15 in the video and another annotation corresponds to seconds 25-30).  I saw better preliminary results by only training with the segments that were in the most agreement with each other (so the 10-15 second annotation in the previous example).  In the json files I provide to train my files (data/tempo{HL/TL}+didemo_{train/val/test}.json) there are two fields; 'times' (used for testing -- this should be consistent with prior work) and 'train_times', which (for the DiDeMo dataset) includes the maxvote time stamp from the ground truth annotations.

### Features

I re-extracted RGB and flow features for my EMNLP experiments, so they are slightly different than the [originally released features](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md).  By running ```./setup.sh```, you will download the features I used to train my models for EMNLP 2018.  Raw fc7 features for both flow and RGB have been released via Google Drive, as well as code to create average pooled features.  Please look at instructions under **Pre-Extracted Features** in the [original DiDeMo release](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md) for more details.

### Getting Raw Videos

I provide pre-extracted video features, but if you are interested in downloadin the raw videos, please see the instructions for downloading videos [here](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md). 
