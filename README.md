# CSC 466 Lab 6 - PageRank

## Group Members
1. Aidan Barbieux - abarbieu@calpoly.edu

## Instructions 

### pageRank.ipynb
    usage: textVectorizer.py [-h] [--stem [STEM]] dir sw_file out

    positional arguments:
    dir            directory containing data
    sw_file        stopword file name (optional)
    out            output file name

    optional arguments:
    -h, --help     show this help message and exit
    --stem [STEM]  stemming, default is False
    --gt [GT]      whether or not to generate groundtruth file  

### pageRank.py
    Usage1 python3 pageRank.py <datafile.>[csv/txt] <dataformat>[SNAP/SMALL] <d>[0-1] <epsilon>[~0.00001]

    positional arguments:
    datafile                .csv or .txt file with data in SNAP or SMALL format given by lab spec
    dataformat                
    optional arguments:
    --output OUTPUT     file name for output; default is out.csv
    --k K               k value for KNN; default is 10
    --m {okapi,cosine}  similarity metric to be used, default is okapi

### classifierEvaluation.py
    usage: classifierEvaluation.py [-h] filename groundtruth output

    positional arguments:
    filename     name of file containing output from RF/KNNAuthorship
    groundtruth  name of csv file containing groundtruth
    output       name for output csv file to contain confusion matrix

    optional arguments:
    -h, --help   show this help message and exit