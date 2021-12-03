#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 6
# Aidan Barbieux - abarbieu@calpoly.edu

import argparse
import numpy as np
import pandas as pd
import sys
import re
from scipy import sparse
from tabulate import tabulate
import matplotlib as mpl
import matplotlib.pyplot as plt
    
def readData(datafile, dataformat):
    """Reads datafile into a list of node pairs representing edges.
    
    Parameters:
    datafile -- the name of the file containing the data
    dataformat -- either 'SMALL' or 'SNAP' representing the data format
    
    Output:
    edges -- 2d list of nodeid pairs
    names -- sorted list of unique node names with index representing id (which may be redundant)
    """
    if dataformat == "SNAP":
        with open(datafile, 'r') as f:
            lines = [[int(node) for node in re.split('\t',edge.strip('\n'))[:2]] for edge in f.readlines() if edge[0][0] != '#']
        edges = np.array(lines)
        return edges, sorted(np.unique(np.array(lines)))
    elif dataformat == "SMALL":
        df = pd.read_csv(datafile, header=None, usecols=[i for i in range(4)])
        if type(df[2][0]) == str:
            df[2] = df[2].str.replace('"', '').str.strip()

        names = sorted(np.unique(np.concatenate((df[0].unique(),df[2].unique()))))

        a = np.array(df[0].apply(names.index))   
        b = np.array(df[2].apply(names.index))
        edges = np.array([b,a]).T
        return edges, names
    else:
        print("dataformat must be 'SMALL' or 'SNAP'")
        exit(1)
    
def createAdjMatrix(edges, names):
    """Creates a scipy lil adjacency matrix from edge list"""
    adj = sparse.lil_matrix((np.max(edges)+1, np.max(edges)+1))
    adj[edges[:,0], edges[:,1]] = 1   
    return adj

def fixSinkNodes(adj):
    """Connects sink nodes in adjacency matrix (in lil form) to themselves """
    degOut = adj.getnnz(axis = 1) # num of non zero values in row

    adj.setdiag(degOut == 0) # more efficient with lilmatrix
    adj = adj.tocsr()
    degOut = adj.getnnz(axis = 1) # num of non zero values in row
    return adj

def scaleAdjMatrix(adj):
    """Scales all edges in adjacency matrix by the nodes out degree"""
    degOut = adj.getnnz(axis = 1) # num of non zero values in row

    degOutRep = np.repeat(degOut, degOut) # degOut is the same as number of data points in row
    adj.data = adj.data / degOutRep
    return adj

def mean_sse(a,b):
    """Calculates sum of square errors on two 1d vectors a and b"""
    if len(a) != len(b):
        print("a and b must be of equal length")
        exit(1)
    return np.sum((a-b)**2) * 1.0/len(a)


def iteratePageRank(adj):
    """Runs pagerank until maximal difference between ranks is minimal, returns number of iterations"""
    p = [1/adj.shape[0]] * adj.shape[0] 
    d = 0.9
    jProb = [(1-d)/len(p)] * len(p)
    maxdiff = 1
    numiters = 0
    while maxdiff > 0.00001:
        numiters+=1
        prevP = p
        p = adj.T*p*d + jProb
        maxdiff = max(abs(prevP-p))
    return p, numiters

def printResults(p, names, n=-1):
    """Prints names and rank of dataset for first n points, -1 is all"""
    n = np.clip(n, -1, len(names))
    nn = 0
    ps = np.column_stack((names,p))
    df = pd.DataFrame(ps[ps[:,1].argsort()][::-1][:n], columns = ["actor", "pagerank"])
    print(tabulate(df, headers='keys', tablefmt='psql'))

def parse():
    parser = argparse.ArgumentParser(description="Page Rank")
    parser.add_argument(
        "datafile", 
        type=str, 
        help=".csv or .txt file with data in SNAP or SMALL format given by lab spec"
    )
    parser.add_argument(
        "dataformat", 
        type=str, 
        choices=["SMALL", "SNAP"],
        help="SMALL or SNAP to determine in which type the data is given"
    )
    parser.add_argument(
        "--d",
        type=float,
        default=0.85,
        help="probability of staying on the same page, usually between 0.7 and 0.95",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00001,
        help="maximal difference between pagerank iterations at which to finish"
    )

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = parse()
    datafile = args["datafile"]
    dataformat = args["dataformat"]
    d = args["d"]
    epsilon = args["epsilon"]
              
    edges, names = readData(datafile, dataformat)
    adj = createAdjMatrix(edges,names)
    adj = fixSinkNodes(adj)    
    adj = scaleAdjMatrix(adj)
    p, numiters = iteratePageRank(adj)
    printResults(p, names)
              
    