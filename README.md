

# L2R Likelihood Estimation in GaP-FA 

## C++ code in a Docker container
To ensure full reproducibility and ease the installation process, we deliver the c++ code in a Docker container. 

To run it you need to install Docker Compose following the instructions [here](https://docs.docker.com/compose/install/).

To run the code just clone the repository.
On the following instructions you are always assumed to be in the root directory of the repository (the directory where the docker-compose.yml file is).

### Definition of docker-compose environment variables
Do `make-env.sh`
<!--The file env.sh contains the definition of the docker-compose environment variables. It can be customized -->
### Build up the docker container
Do `docker-compose build`. This will take a while, because it will build the container that will run the code.
### Compile the C++ code
Do `docker-compose run pfa bin/compile.sh`
After that, to speed up the process you can also perform partial compilations for example, to rebuild and install just the pfa library you can do 
`docker-compose run -w /pfa/build pfa make pfa-install`
### Execute a C++ executable file
Say you want to run the test-script executable. You do `docker-compose run pfa bin/test-script`
After that you can edit the code on your machine and repeat the compilation when you are done.
### Opening a terminal @ the docker container
To connect to the docker container do `docker-compose run pfa bash`
### Launching a python script at the docker container
From inside the container, you can launch python scripts easily
* If they are executable and correctly marked with the `#!/usr/bin/env python3` as its first line: Do `docker-compose run pfa python/exact.py Twitter 10653 100 5 10 1e9 4` 
* Otherwise: `docker-compose run pfa /usr/bin/env python3 python/exact.py Twitter 10653 100 5 10 1e9 4` 


## Python Binding

We have created a Python binding of the C++ code in order to give more accessibility to the code. 

To import the estimation method library, simply import the library:
```python
import pfa
```

This library contains the following methods:

```python
y_D, p, r, Phi = pfa.load_model(filename, N, W, K)
pfa.inference(method, opts, y, Phi, r, p)
pfa.inference_ds(method, opts, y_D, Phi, r, p)
```

The `load_model()` method loads the model `Phi, r, p` and dataset `y_D` specified in `filename=data/Twitter_GaP_10653N_100W_5T.mat` with `N=10653`, `W=100` and `K=5`.

The `inference()` method calculates or estimates the marginal document likelihood for a single document `y`  and a model `Phi, r, p` as per the estimation `method` specified with a string `Exact` `DS` `HM` `L2R` or `L2R`.

The `inference_ds()` method does the same than `inference()`, but for set of documents `y_D`.

`opts` is a json string which defines the sampler options `"{\"num_threads\" :" + str(num_threads) + ", \"num_partials\":" + str(num_partials) + ", \"num_samples\":" + str(num_samples) + "}"`


## Python scripts

We deliver this code together with a set of python scripts which are helpful to reproduce the experiments in the paper:
*A Left-to-right Algorithm for Likelihood Estimation in Gamma-Poisson Factor Analysis*

After downloading the GaP models and collections from [here](https://doi.org/10.7910/DVN/GDTAAC) to `data/` folder, we can use the scripts in `python/` folder to estimate or calculate the marginal likelihoods for both dimensioned and realistic scenarios. 

- To run the experiments in reasonably sized scenarios, make sure first that `[DATASETNAME]_GaP_\*5T.mat` files are in the `data` folder. Then, do:
```python
docker-compose run pfa python/exact.py Twitter 10653 100 5 1000 1e9 4 
```
where the 10653 corresponds to the number of documents, 100 is the vocabulary size, 5 the maximum number of topics, 1000 the number of documents to evaluate its marginal document likelihood, 1e9 is the maximum number of sums allowed in the formula of the exact marginal and 4 are the number of threads. 

- To run the experiments in realistic scenarios, make sure first that `[DATASETNAME]_GaP_\*100T.mat` files are in the `data` folder. Then, do:
```python
docker-compose run pfa python/estimate_DS.py Twitter 10523 6344 100 1000 0 4 
```
where the maximum number of sums is set to 0, to indicate that we do not filter out documents with intractable marginal and all other parameters play the same role.
