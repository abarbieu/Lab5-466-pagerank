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

    def __init__(self, datafile, datatype, d=0.9, iters=30):
        self.datafile = datafile
        if datatype != "SMALL" and datatype != "SNAP":
            print("datatype must be 'SMALL' or 'SNAP'")
            exit(1)
        self.datatype = datatype
        self.d = d
        self.iters = iters
    
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
        return edges, names
    else:
        print("dataformat must be 'SMALL' or 'SNAP'")
        exit(1)
    
    def createAdjMatrix(edges, names):
        """Creates a scipy lil adjacency matrix from edge list"""
        adj = sparse.lil_matrix((len(names), len(names)))
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
        
        
    def nextPageRank(adj, jumpProbVect, prevPageRank=None):
        """Runs pagerank on adjacency matrix adj
        
        Parameters:
        adj -- adjacency matrix (scaled by out degree, sink nodes fixed)
        jumpProbVect -- vector of jump probabilities for simplicity/efficiency: [(1-d)/len(p)] * len(p)
        prevPageRank -- for any iteration > 0: previous output, otherwise it will be instantiated
        
        Output:
        p -- p^k+1 pageranks (index is nodeid)
        maxdiff -- maximal difference between one pagerank and the next
        """
        if prevPageRank is None:
            p = [1/adj.shape[0]] * adj.shape[0] 
        
        
    def iteratePageRank(self):
        """Runs pagerank iter number of times"""
        d = 0.9
        jumpProbVect = [(1-d)/len(p)] * len(p) # vector of jump probabilities (uniform)
        maxdiffs = []
        for i in range(100):
            prevP = p
            p = adj.T*p*d + jProb
            maxdiffs.append(max(abs(prevP-p)))

    def inducePageRank(self):
        self.readData()
        self.createAdjMatrix()
        self.fixSinkNodes()
        self.scaleAdjMatrix()
        return self.iteratePageRank()