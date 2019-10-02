#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from utils import stampTime, tabWriter, tabReader, readORrun, objectContainer

def __main__():
    print 'This module is not intended for direct command line usage. Currently supports import to Python only.'
    moduleTest()
    return

def moduleTest():
    pr = projectionData('hi')
    bg, coord = pr.readManifold('testdata/projection.csv')
    pr.createAnnotation('testdata/test_annotation.tsv', coordict=coord)
    return

class manyMani(objectContainer):
    def add(self, name, **kwargs):
        self.order.append(name)
        a = locals()
        a = a['kwargs']
        self.data[name] = projectionData(name, **a)
        return

class projectionData:
    def __init__(self, name, coordinates={}, annotations={}, barcodecollision={}, gex=None, matrix=None, labels=None, projectRoot='', proFile=None, coordFile=None, annotFile=None, conflictresolve='drop', keepextra=True, numCellType=12, numCores=8, force_recompute=False, verbose=False):
        self.name = name
        self.label = name
        self.verbose=verbose
        if gex != None:
            matrix = gex.matrix
            if labels == None:
                labels = gex.cells
            if proFile != None:
                self.background, self.coordinates, self.annotations = readORrun(proFile, force_recompute, self.readAnnMan, self.createAnnMan, [[[], []], {}, {}], parentDir=projectRoot, kwargs={'matrix': matrix, 'labels':labels, 'numCellType': numCellType, 'numCores': numCores})
            else:
                self.background, self.coordinates = readORrun(coordFile, force_recompute, self.readManifold, self.createManifold, [[[], []], {}], parentDir=projectRoot, kwargs={'matrix': matrix, 'labels':labels})
                self.annotations = readORrun(annotFile, False, self.readAnnotation, self.createAnnotation, {}, parentDir=projectRoot, kwargs={'coordict': self.coordinates, 'numCellType': numCellType, 'numCores': numCores})
        else:
            if proFile != None:
                self.background, self.coordinates, self.annotations = self.readAnnMan(projectRoot+proFile)
            else:
                self.background, self.coordinates = self.readManifold(projectRoot+coordFile)
                self.annotations = readORrun(annotFile, False, self.readAnnotation, self.createAnnotation, {}, parentDir=projectRoot, kwargs={'coordict': self.coordinates, 'numCellType': numCellType, 'numCores': numCores})
        if len(barcodecollision) > 0:
            self.coordinates, self.annotations = self.recodeBarcodes(barcodecollision, conflictresolve)
        cellTypes = {}
        for k, v in self.annotations.iteritems():
            if v not in cellTypes:
                cellTypes[v] = []
            cellTypes[v].append(k)
        self.cellTypes = cellTypes
        return
    
    def recodeBarcodes(self, bc, conflictresolve):
        cb = {}
        for k, v in bc.iteritems():
            s = set(v)
            if len(s) == 1:
                s = list(s)
                cb[s[0]] = k
            else:
                if conflictresolve == 'keepall':
                    for a in s:
                        cb[a] = k
        coordinates, annotations = {}, {}
        for k, v in self.coordinates.iteritems():
            if k in cb:
                coordinates[cb[k]] = v
            else:
                coordinates[k] = v
        for k, v in self.annotations.iteritems():
            if k in cb:
                annotations[cb[k]] = v
            else:
                annotations[k] = v
        if self.verbose:
            stampTime('Barcodes recoded to match experimental layout.')
        return coordinates, annotations 
                
    def createManifold(self, fn, matrix=None, labels=[], save_output=True):
        bg, coord = [[], []], {}
        if matrix == None:
            return bg, coord
        else:
            X = matrix.transpose()
            X = X.toarray()
            X_embedded = [[0, 0], [0, 0]]
            X_embedded = TSNE(n_components=2).fit_transform(X)
        coord = dict(zip(labels,X_embedded))
        if save_output:
            tabWriter(X_embedded, fn, rownames=labels, tabformatter='list_of_list')
        X_embedded = zip(*X_embedded)
        if self.verbose:
            stampTime('The manifold ' + self.name + ' was calculated.')
        return X_embedded, coord
    
    def readManifold(self, fn, matrix=None, labels=None, save_output=True):
        coord = tabReader(fn, tabformat='dict_of_list', floating=[1, 2], add_nan=False)
        ncoord = {}
        for k, v in coord.iteritems():
            if k[-2] != '-':
                ncoord[k+'-1'] = v
            else:
                ncoord[k] = v
        coord = ncoord
        bg = [[], []]
        if labels == None:
            labels = list(coord.keys())
        for i in range(0, len(labels)):
            bg[0].append(coord[labels[i]][0])
            bg[1].append(coord[labels[i]][1])
        if self.verbose:
            stampTime('The manifold ' + self.name + ' was read from file.')
        return bg, coord
    
    def createAnnotation(self, fn, coordict={}, numCellType=12, numCores=8, save_output=True):
        Y, X = zip(*coordict.items())
        labels = SpectralClustering(n_clusters=numCellType, n_jobs=numCores, assign_labels="discretize", random_state=0).fit_predict(X)
        annotations = {}
        for i in range(len(labels)):
            annotations[Y[i]] = 'Cell type ' + str(labels[i])
        if save_output:
            tabWriter(annotations, fn)
        if self.verbose:
            stampTime('The manifold ' + self.name + ' was read from file.')
        return annotations
    
    def readAnnotation(self, fn, coordict={}, numCellType=12, numCores=8, save_output=True):
        annotations = tabReader(fn, tabformat='dict_of_list', evenflatter=True)
        if self.verbose:
            stampTime('Annotation of the manifold ' + self.name + ' was read from file.')
        return annotations
    
    def createAnnMan(self, fn, matrix=None, labels=[], numCellType=12, numCores=8):
        bg, coord = self.createManifold(fn, matrix=matrix, labels=labels, save_output=False)
        annotations = self.createAnnotation(fn, coordict=coord, numCellType=numCellType, numCores=numCores, save_output=False)
        odict = {}
        for k, v in annotations.iteritems():
            odict[k] = [v]+coord[k]
        tabWriter(odict, fn)
        if self.verbose:
            stampTime('Annotation and coordinates for the manifold ' + self.name + ' were created and written into a single file.')
        return bg, coord, annotations
    
    def readAnnMan(self, fn, matrix=None, labels=[], numCellType=12, numCores=8):
        bg, coord, annotations  = [[], []], {}, {}
        data = tabReader(fn, tabformat='dict_of_list', floating=[1, 2], add_nan=False)
        for cell, dat in data.iteritems():
            x, y, l = dat
            annotations[cell] = l
            coord[cell] = [x, y]
            bg[0].append(x)
            bg[1].append(y)
        if self.verbose:
            stampTime('Annotation and coordinates for the manifold ' + self.name + ' were read from file.')
        return bg, coord, annotations
  
if __name__ == '__main__':
    __main__()