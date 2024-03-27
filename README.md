# SpikeNet Benchmarking
NOTE: This repository is NOT the original, you can find that here: https://github.com/EdisonLeeeee/SpikeNet?tab=readme-ov-file, these files have been copied over purely to benchmark this method on new datasets!

For the dblp data, dblp.txt contains the edges + timesteps and the label.txt contains the labels for each node, it's pretty straightforward. Each dataset has its own dataloader class since each dataset is stored in a different way (patent uses json files).