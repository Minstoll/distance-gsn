# Biconnectivity and Zinc

Codebase for cut edge and cut node detection on a synthetic dataset, as well as regression on the zinc dataset, with and without resistance distances.

Code for generating the biconnectivity (cut edge and cut node detection) dataset is found in biconn_data_gen.ipynb.

Hombasis-gnn is a modified clone of https://github.com/ejin700/hombasis-gnn, which contains the zinc dataset among others. 

The procedure for running experiments is the same as that in hombasis-gnn, and the run_experiments.bat file can be modified to run the desired sequence of experiments. Everything else to do with biconnectivity tests and tests on zinc with resistance distances is in hombasis-gnn/hombasis-bench, i.e. the remainder of hombasis-gnn is as the original.

We used 140 epochs on ZINC and 60 epochs on biconn. The results are averaged over the seeds 41, 95, 12, 35.
