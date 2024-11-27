# Preprocessing Datasets for REGENT

## Offline Retrieval
First, we iterate over all states in the pre-training data used by REGENT (where each states gets a turn to be a query state). 
As output, we get the indices of the states in the context for each query state. We call this output the retrieved indices.

```
python -u scripts_regent/offline_retrieval.py --tasks ${TASK}
```
Here, you can give the whole domain name (metaworld, mujoco, babyai) as TASK instead of each individual environment. For speedy runs, we recommend giving and running the 57 individual environment names in atari in parallel.


## New Dataset Bin File Creation 
Second, we compute the distances between every state (retrieved and query) and the first retrieved state.
The distances are (a) L2 distances between raw vector observations, or (b) L2 distance between atari embeddings. 
The distances are normalized by their p95 values and used during pre-training.
We also take the subset of the JAT dataset that we actually use and save everything (states, rewards, actions) as bin files.
For training environments, this subset has enough demos for 100k states per environment. 
For unseen environments, we take the small number of demos available for retrieval. 
We use them only to calculate the distances whose p95 value is needed for normalizing diatances computed at eval time.

```
python -u scripts_regent/new_dataset_bin_files_creation.py --task ${TASK}
```
Same as above, consider giving domain name (except in atari).


## New Dataset Embeddings Creation (for Atari only)
Third, we save all embeddings of atari images also as a bin file.

```
python -u scripts_regent/new_dataset_embeddings_creation.py --task ${TASK}
```


## Push to HF
We combine the bin files for the states, actions, rewards, and embeddings (if atari) and push to the `task_subset` folders on HF. 
We combine the (retrieved) indices and distances and push to the `task_newdata` folders on HF.

```
python scripts_preprocessing/convert_bin_files_to_parquet_files_and_push_to_hf.py
```

## Changes to be made for REGENT with cosine distance
The aforementioned changes are as follows.
* Please change `metric_type='l2'` to `metric_type='ip'` on line 606 of [regent/utils.py](regent/utils.py). This sets the retrieval index distance metric to inner product.
* Uncomment line 596 in [regent/utils.py](regent/utils.py). This normalizes the embeddings/observations since cosine similarity is the inner product of normalized embeddings/observations.
* Please change `return np.sqrt(np.sum((a - b)**2))` to `return 1.0-cosine_similarity(a,b)` on line 290 of [regent/utils.py](regent/utils.py). This is essential for computing the distance values and simply replaces the l2 distance with cosine distance.

Then follow all of the above instructions to redo the processing of the training dataset with cosine distance instead of l2.
 
