# Defect Detection Benchmark

Work in progress framework for benchmarking vulnerability/defect detection models. In my travels
around this area of research, there are many different datasets and models used. Some have done
benchmarking on several datasets and models. However, it adds a bunch of wasted time to every
research project in the space if we all have to duplicate this work. 

This project glues together currently 8 different datasets of different formats and specifications
into one consistent jsonl format read for use with the model training, testing and inference code.
The base of that code is from `https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection` though it has been edited and expanded to use as a more flexible framework.

This is designed to be forked and extended by the user. It is a base to work from. It is not trying
to be an all encompassing framework with many features. If you need something, generally you will
need to edit the code itself. The hope is that as it is small and mostly contained this will be
easy.

## Datasets

Currently placeholder descriptions
```
MVDSC Mixed -> Various formats, some in AST/Graph form, some of juliet separated into one function per file
MVDSC -> Various formats, some in AST/Graph form, some of juliet separated into one function per file | seems like the raw files aren't in there?
Taxonomy of Buffer Overflows -> very synthetic, one folder per c example, detailed classification
Stonesoup -> One folder per example, very noisy examples with extra stonesoup code for tracing, mix of c and java?
Juliet c/c++ -> Nested directories, fault and fix in same file, very synthetic
Devign -> Qemu and ffmpeg functions with errors, json with difficult realworld code
Draper -> Hdf5 files, unsure yet but I believe large aggregation of c/c++ realworld code from github scraping
VulDeepecker -> From SARD dataset, c/c++, similar to juliet but only buffer and resource management mix of synthetic and real?CGD format?
DiverseVul -> Realworld json format from git commits, c/ c++?
Reveal -> Some cleaned Devign? Chrome and debian real world dataset in json format
ICVul -> Collection of vulnerability contributing commits from CVEs linked to github commits, realworld csv
VulBERTa MLP Devign -> Collection of many of the above datasets in there own format
```

# Usage

```bash
#Clone or fork this repo
https://github.com/ijakenorton/defect_detection_benchmark

#Setup enviroment. Currently I am using conda.
conda env create -f environment.yml
#There may need to be some messing around with the environment depending on versions. The
#environment has been tested on for Rocky Linux 9.2 (Blue Onyx) & Pop-os  

#Not currently implemented but the idea is to run
#This is still in flux as the dataset locations need to be stabilised
sh ./data/get_data.sh

#The only model which needs external download is the natgen pretrain model.
sh ./models/get_models.sh

#Then to train the suite
cd scripts
sh ./train_all.sh

#The results will be output to the same place as the model output path. From the root of that
#directory ../../scripts/aggregate_results.sh can be run to show the results from that model

```



# Constraints

Currently I run all the models on H100s with 64gb of RAM. I believe most of the datasets will not
need such a heavy duty setup. Draper and Diversevul are very large datasets and so will most likely
be more difficult to run on smaller GPUS. Modifying the batch sizes may help this though

I intend to do some testing on what are the minimal specs required for each of the default training options
