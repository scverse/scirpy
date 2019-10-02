#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import stampTime, tabWriter, tabReader, readORrun, objectContainer
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import numpy as np
import scipy.cluster.hierarchy as hcluster
import scipy.spatial as ssd


def __main__():
    print 'This module is not intended for direct command line usage. Currently suports import to Python only.'
    return

class distList(objectContainer):
    def add(self, *args, **kwargs):
        a = locals()
        name = a['args'][0]
        self.order.append(name)
        self.data[name] = distMat(*a['args'], **a['kwargs'])
        return

class distMat:
    def __init__(self, name, cells, clonotypes, metric='TCR', projectRoot='', matrixFile=None, maxGroupNum=10, penalties=[-10, -2], include_secondary_chains=True, force_recompute=False, verbose=False):
        self.not_kidera = True
        self.clabels = None
        self.label = name
        self.verbose = verbose
        self.include_secondary_chains=include_secondary_chains
        if metric == 'Kidera':
            self.not_kidera = False
            self.clabels = ['K'+str(x) for x in range(1, 11)]
            matrixFile = matrixFile+'_matrix_kidera.tsv'
            self.forheat, self.distances, self.labels = self.makeDisKid(cells)
        else:
            matrixFile = matrixFile+'_matrix_tcr.tsv'
            self.distances, self.labels = readORrun(matrixFile, force_recompute, self.readDisTCR, self.runDisTCR, [[], []], parentDir=projectRoot, kwargs={'cells': cells, 'clonotypes': clonotypes, 'penalties': penalties, 'secondary': self.include_secondary_chains})
            self.forheat = self.distances
        self.labels = tuple(self.labels)
        self.disDic = self.makeDisDic()
        self.dendro, self.linkage = self.dendroFromDist(self.distances, self.labels)
        self.treeGroups = readORrun(matrixFile[:-4]+'_annotation.tsv', force_recompute, self.readTreeGroups, self.findTreeGroups, {}, kwargs={'linkage': self.linkage, 'labels': self.labels, 'maxGroupNum': maxGroupNum})
        return
    
    def runDisTCR(self, fn, cells, clonotypes, penalties, secondary):
        clones = {}
        matrix = matlist.blosum62
        gap_open, gap_extend = penalties
        for clone, info in clonotypes.iteritems():
            cdrs = info['cdrSym']
            if cdrs != '|||':
                tcrs = []
                cdrs = cdrs.split('|')
                for i in range(0, len(cdrs)):
                    aa = cdrs[i]
                    if aa == '':
                        aa = 'XXXXXXXXXXXX'
                    cdrs[i] = aa
                tcrs.append(cdrs[:2])
                if secondary:
                    if cdrs[2] + cdrs[3] != 'XXXXXXXXXXXXXXXXXXXXXXXX':
                        tcrs.append(cdrs[2:4])
                clones[clone] = tcrs
        similarities = {}
        selfsimilarities = {}
        ldict = {}
        cl = list(clones.keys())
        for c1 in cl:
            for c2 in cl:
                clonescore = []
                for a1, b1 in clones[c1]:
                    for a2, b2 in clones[c2]:
                        similarity_score = 0
                        similarity_score += pairwise2.align.globalds(a1, a2, matrix, gap_open, gap_extend, one_alignment_only=True, score_only=True)
                        similarity_score += pairwise2.align.globalds(b1, b2, matrix, gap_open, gap_extend, one_alignment_only=True, score_only=True)
                        clonescore.append(similarity_score)
                clonescore = np.max(clonescore)
                for cell1 in clonotypes[c1]['cells']:
                    ldict[cell1] = c1
                    if cell1 not in selfsimilarities:
                        selfsimilarities[cell1] = -10000
                    if clonescore > selfsimilarities[cell1]:
                        selfsimilarities[cell1] = clonescore
                    if cell1 not in similarities:
                        similarities[cell1] = {}
                    for cell2 in clonotypes[c2]['cells']:
                        if cell2 not in selfsimilarities:
                            selfsimilarities[cell2] = -10000
                        if clonescore > selfsimilarities[cell2]:
                            selfsimilarities[cell2] = clonescore
                        if cell2 not in similarities:
                            similarities[cell2] = {}
                        if cell1 not in similarities[cell2]:
                            similarities[cell2][cell1] = []
                        if cell2 not in similarities[cell1]:
                            similarities[cell1][cell2] = []
                        similarities[cell1][cell2].append(clonescore )
                        similarities[cell2][cell1].append(clonescore )
        distances = []
        labels = list(similarities.keys())
        #labels.sort(key=lambda x: ldict[x]) #Uncomment if need to order clonotypes. Not very useful since renaming from cell ranger clonotype names.
        for l1 in labels:
            distance_row = []
            for l2 in labels:
                if ldict[l1] == ldict[l2]:
                    distance_row.append(0.0)
                else:
                    distance_row.append(1-(np.mean(similarities[l1][l2])/selfsimilarities[l1]))
            distances.append(distance_row)
        #This part contains a bit of cosmetics to make a nice, simmetric matrix. Most of if might become pointless later
        distances1 = np.array(distances)
        distances2 = np.rot90(np.flipud(distances1), 3)
        distances = np.mean(np.array([distances1, distances2]), axis=0)
        tabWriter(distances, fn, lineformatter=lambda y: ['{:.5f}'.format(x) for x in y], tabformatter='list_of_list', colnames=labels, rowname='')
        if self.verbose:
            stampTime('Computed TCR distances and saved to ' + fn)
        return distances, labels
    
    def readDisTCR(self, fn, cells, clonotypes, penalties, secondary):
        def lineParse(data, line, lineNum, firstline, includend, test_only=False):
            if test_only:
                return locals()
            if firstline:
                firstline = False
                data[1] = line[1:]
            else:
                data[0].append([float(x) for x in line])
            return data, firstline, includend
        distances, labels = tabReader(fn, tabformatter=lambda x: [[], []], lineformatter=lineParse)
        if self.verbose:
            stampTime('Read TCR distances for ' + str(len(labels)) + ' cells from ' + fn)
        return np.array(distances), labels
    
    def makeDisKid(self, cells):
        features, labels = [], []
        for k, v in cells.iteritems():
            d, validator = [], True
            eli = v['kideras']['C']
            if 'NaN' in eli:
                validator = False
            else:
                for e in eli:
                    try:
                        d.append(float(e))
                    except:
                        validator = False
            if validator:
                features.append(d)
                labels.append(k)
        distances = ssd.distance.pdist(features, 'minkowski')
        #distances = ssd.distance_matrix(features, features)
        distances = ssd.distance.squareform(distances)
        if self.verbose:
            stampTime('A Kidera-based distance matrix is computed.')
        return np.array(features), distances, labels
    
    def makeDisDic(self):
        d = {}
        distances, labels = self.distances, self.labels
        N = len(labels)
        for i in range(0, N):
            for j in range(0, N):
                c = [labels[i], labels[j]]
                c.sort()
                c = '|'.join(c)
                d[c] = distances[i][j]
        if self.verbose:
            stampTime('Quick access distance dictionary created.')
        return d
    
    def dendroFromDist(self, distances, labels, ax=None):
        #distances = ssd.distance.squareform(distances) #This would switch between condensed and redundant matrix formats. Currently produces error beacuse it finds negative distances.
        linkage = hcluster.linkage(distances)
        dendro = hcluster.dendrogram(linkage, labels=labels, ax=ax, distance_sort=True, link_color_func=lambda k: 'black')
        if self.verbose:
            stampTime('Dendrogram created based on distance matrix.')
        return dendro, linkage
    
    def readTreeGroups(self, fn, linkage, labels, maxGroupNum):
        members = tabReader(fn, tabformat='dict_of_list')
        if self.verbose:
            stampTime('Groups of distance matrix read from file' + fn)
        return members
    
    def findTreeGroups(self, fn, linkage, labels, maxGroupNum):
        groups = hcluster.fcluster(self.linkage, maxGroupNum, criterion='maxclust')
        members = {}
        for i in range(0, len(groups)):
            member = 'Group ' + str(groups[i])
            if member not in members:
                members[member] = []
            members[member].append(labels[i])
        tabWriter(members, fn, lineformatter=lambda x: x, tabformatter='dict_of_list')
        if self.verbose:
            stampTime('Groups isolated from distance matrix.')
        return members
    
    def meanDistances(self, groups, order):
        disDic = self.disDic
        labels = self.labels
        distances = []
        for name1 in order:
            group1 = groups[name1]
            distancerow = []
            for name2 in order:
                group2 = groups[name2]
                distance = []
                for c1 in group1:
                    if c1 in labels:
                        for c2 in group2:
                            if c2 in labels:
                                kl = [c1, c2]
                                kl.sort()
                                c = '|'.join(kl)
                                if c in disDic:
                                    distance.append(disDic[c])
                if len(distance) > 0:
                    distancerow.append(np.mean(distance))
                else:
                    distancerow.append(1.0)
            distances.append(distancerow)
        return distances
    
    def groupTree(self, groups, order=None, ax=None):
        if order == None:
            order = list(groups.keys())
        distances = self.meanDistances(groups, order)
        tree = self.dendroFromDist(distances, order, ax=ax)
        return tree
    
    def biClustMap(self, ax1=None, ax2=None):
        l_rows = hcluster.linkage(self.forheat)
        l_cols = hcluster.linkage(self.forheat.T)
        v_rows, v_cols = hcluster.leaves_list(l_rows), hcluster.leaves_list(l_cols)
        t1 = hcluster.dendrogram(l_rows, ax=ax1, no_labels=True, distance_sort=True, link_color_func=lambda k: 'black', orientation='left')
        t2 = hcluster.dendrogram(l_cols, ax=ax2, no_labels=self.not_kidera, distance_sort=True, link_color_func=lambda k: 'black')
        data = self.forheat[v_rows, :]
        data = data[:, v_cols]
        return data, t1, t2

if __name__ == '__main__':
    __main__()