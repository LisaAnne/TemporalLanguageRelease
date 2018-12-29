
# Localizing Moments in Video with Temporal Language.

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

## Data

Please see our TEMPO Human Language and Template Language annotations [here](https://people.eecs.berkeley.edu/~lisa_anne/tempo/initial_release_data.zip).

### TEMPO Annotations

The TEMPO annotations will have the following fields:

* annotation_id: Each annotation_id is asigned as {temporal_word}\_{didemo_id} where the temporal_word is "before", "after", "then" or (in TEMPO-HL) "while".  The didemo_id is the didemo id for the context moment. 
* context: The ground truth context for the sentence.
* description
* reference_description: Reference description given to the AMT workers.  Note that we did not provide reference descriptions when collecting "while" annotations.  For "then" annotations in TEMPO-TL there will be two reference descriptions.
* times: Three ground truth human annotations.  All three are used in evaluation.  See the paper for details.
* video
