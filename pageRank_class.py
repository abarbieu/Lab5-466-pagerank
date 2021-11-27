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

class PageRank:
    def __init__(self, datafile, datatype, d=0.9, iters=30):
        self.datafile = datafile
        if datatype != "SMALL" and datatype != "SNAP":
            print("datatype must be 'SMALL' or 'SNAP'")
            exit(1)
        self.datatype = datatype
        self.d = d
        self.iters = iters
    
    def readData(self):
        if datatype == "SNAP":
            with open(self.datafile, 'r') as f:
                lines = [[int(node) for node in re.split('\t',edge.strip('\n'))[:2]] for edge in f.readlines() if edge[0][0] != '#']
            self.edges = np.array(lines)
            self.numItems = len(np.unique(np.array(lines)))
        elif datatype == "SMALL":
            df = pd.read_csv(self.datafile, header=None, usecols=[i for i in range(4)])
            if type(df[2][0]) == str:
                df[2] = df[2].str.replace('"', '').str.strip()

            self.names = sorted(np.unique(np.concatenate((df[0].unique(),df[2].unique()))))

            a = np.array(df[0].apply(self.names.index))   
            b = np.array(df[2].apply(self.names.index))
            self.edges = np.array([b,a]).T
            self.numItems = len(names)
        else:
            print("datatype must be 'SMALL' or 'SNAP'")
            exit(1)
    
    def createAdjMatrix(self):
        self.adj = sparse.lil_matrix((self.numItems, self.numItems))
        self.adj[self.edges[:,0], self.edges[:,1]] = 1    
    
    def fixSinkNodes(self):
        # connects sink nodes to themselves
        degOut = self.adj.getnnz(axis = 1) # num of non zero values in row

        self.adj.setdiag(degOut == 0) # more efficient with lilmatrix
        self.adj = self.adj.tocsr()
    
    def scaleAdjMatrix(self):
        # scales all edges by degree out
        degOut = self.adj.getnnz(axis = 1) # num of non zero values in row
        degOutRep = np.repeat(degOut, degOut) # degOut is the same as number of data points in row
        self.adj.data /= degOutRep
        
    def iteratePageRank(self):
        p = [1/self.adj.shape[0]] * self.adj.shape[0] 
        d = 0.9
        jProb = [(1-d)/len(p)] * len(p)
#         maxdiffs = []
        for i in range(self.iters):
#             prevP = p
            p = self.adj.T*p*d + jProb
#             maxdiffs.append(max(abs(prevP-p)))
        return p

    def inducePageRank(self):
        self.readData()
        self.createAdjMatrix()
        self.fixSinkNodes()
        self.scaleAdjMatrix()
        return self.iteratePageRank()