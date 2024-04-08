# SpikeNet Benchmarking
We evaluate the SpikeNet technique on:
- [TGBL - Review](https://tgb.complexdatalab.com/docs/leader_linkprop/#tgbl-review-v2): The considered task for this dataset is to predict which product a user will review at a given time.
- [TGBL - Genre](https://tgb.complexdatalab.com/docs/leader_nodeprop/#tgbn-genre): The considered task for this dataset is to rank with which set of music genres a user will interact the most over the course of the next week.

# References
- SpikeNet Implementation: https://github.com/EdisonLeeeee/SpikeNet?tab=readme-ov-file
- TGB Implementation: https://github.com/shenyangHuang/TGB/tree/main

# SpikeNet Notes
Each SpikeNet dataset necessitates a unique data loader class to accommodate the distinct data storage formats, such as DBLP using text files for edges, timesteps, and labels, while Patent relies on JSON files.