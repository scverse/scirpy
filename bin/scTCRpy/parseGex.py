#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import tabReader, stampTime

import os, h5py
import numpy as np
from scipy import sparse

def __main__():
    kw = sys.argv
    matrixfn = kw[1]
    ardi = {}
    for n in kw[2:]:
        k, v = n.split('=')
        ardi[k] = v
    if 'projectRoot' in ardi:
        matrixfn = ardi['projectRoot'] + '/' + matrixfn
        del ardi['projectRoot']
    gexMat(matrixfn, **ardi)
    return

class allGex:
    def __init__(self, verbose=False):
        self.verbose=verbose
        self.gex = None
        self.genranks = {}
        self.genorm = {}
        self.barcodehistory={}
        return
    
    def add(self, newMatrix):
        self.barcodehistory=newMatrix.barcodehistory
        if self.gex == None:
            self.gex = newMatrix
        else:
            #merge the intersection first
            g1 = set(self.gex.genes)
            g2 = set(newMatrix.genes)
            present_in_both = g1&g2
            blist = []
            for g in present_in_both:
                blist.append([self.gex.genInd[g], g])
            blist.sort(key=lambda x: x[0])
            indices, genes = zip(*blist)
            upperMat = self.gex.matrix[indices,:]
            lowerMat = newMatrix.matrix[indices,:]
            newgenes = genes[:]
            newalts = list(self.gex.alts[_] for _ in indices)
            todo = g1-g2
            if len(todo) > 0:
                #add an upper wing
                blist = []
                for g in todo:
                    blist.append([self.gex.genInd[g], g])
                blist.sort(key=lambda x: x[0])
                indices, genes = zip(*blist)
                putUp = self.gex.matrix[indices,:]
                putDown = sparse.csc_matrix((len(indices),newMatrix.Ncells))
                upperMat = sparse.vstack([upperMat, putUp])
                lowerMat = sparse.vstack([lowerMat, putDown])
                newgenes += genes
                newalts += list(self.gex.alts[_] for _ in indices)
            todo = g2-g1
            if len(todo) > 0:
                #add a lower wing
                blist = []
                for g in todo:
                    blist.append([newMatrix.genInd[g]-20, g])
                blist.sort(key=lambda x: x[0])
                indices, genes = zip(*blist)
                putUp = sparse.csc_matrix((len(indices),self.gex.Ncells))
                putDown = newMatrix.matrix[indices,:]
                upperMat = sparse.vstack([upperMat, putUp])
                lowerMat = sparse.vstack([lowerMat, putDown])
                newgenes += genes
                newalts += list(newMatrix.alts[_] for _ in indices)
            self.gex.matrix = sparse.hstack([upperMat, lowerMat])
            self.gex.alts = newalts
            self.gex.genes = newgenes
            self.gex.cells += newMatrix.cells
            self.gex.Ngenes = len(self.gex.genes)
            self.gex.Ncells = len(self.gex.cells)
            self.gex.makeIndices()
            self.genranks.update(newMatrix.rankInCell)
            for k, v in newMatrix.genMax.iteritems():
                if k in self.genorm:
                    other = self.genorm[k]
                    if v > other:
                        self.genorm[k] = v
                else:
                    self.genorm[k] = v
        return

class gexMat:
    def __init__(self, fn, genome='matrix', sample_prefix='', mincell=5, minread=0, rankedgenes=100, genenames=True, keeponlygenes=None, filtered=False, barcodehistory={}, matrixfn=None, ranksfn=None, force_recompute=False, verbose=False):
        self.verbose=verbose
        self.barcodehistory=barcodehistory
        self.genome=genome
        self.readMatrix(fn, genenames=genenames, genome=genome)
        if sample_prefix != '':
            self.renameCells(sample_prefix, self.barcodehistory)
        if keeponlygenes == None:
            keeponlygenes = []
        else:
            if not isinstance(keeponlygenes, (list)):
                try:
                    keeponlygenes = tabReader(keeponlygenes, tabformat='list_of_list', evenflatter=True)
                except:
                    keeponlygenes = []
        if len(keeponlygenes) > 0:
            self.throwGenesAway(keeponlygenes)
        else:
            if filtered:
                self.filterMatrix(mincell=mincell, minread=minread)
                if matrixfn != None:
                    if os.path.isfile(matrixfn):
                        if force_recompute:
                            if genenames:
                                self.saveMatrix(matrixfn, self.matrix, self.genome, self.cells, self.genes, self.alts)
                            else:
                                self.saveMatrix(matrixfn, self.matrix, self.genome, self.cells, self.alts, self.genes)
                    else:
                        if os.access(os.path.dirname(matrixfn), os.W_OK):
                            if genenames:
                                self.saveMatrix(matrixfn, self.matrix, self.genome, self.cells, self.genes, self.alts)
                            else:
                                self.saveMatrix(matrixfn, self.matrix, self.genome, self.cells, self.alts, self.genes)
        self.makeIndices() #It is important to create indexes only after we don't plan more modifications
        if ranksfn == None:
            self.rankGenes(minread=minread, rankedgenes=rankedgenes)
        else:
            if os.path.isfile(ranksfn):
                if force_recompute:
                    self.rankGenes(minread=minread, rankedgenes=rankedgenes)
                    self.saveRanks(ranksfn)
                else:
                    self.readRanks(ranksfn)
            else:
                self.rankGenes(minread=minread, rankedgenes=rankedgenes)
                if os.access(os.path.dirname(ranksfn), os.W_OK):
                        self.saveRanks(ranksfn)
        return
    
    def readMatrix(self, fn, genenames=True, genome='matrix'):
        f = open(fn)
        f.close()
        with h5py.File(fn, 'r') as f:
            ds = f[genome]
            if genenames:
                genes = tuple(ds['features']['name'])
                alts = tuple(ds['features']['id'])
            else:
                genes = tuple(ds['features']['id'])
                alts = tuple(ds['features']['name'])
            barcodes = tuple(ds['barcodes'])
            shape = tuple(ds['shape'])
            data = np.array(ds['data'])
            indices = np.array(ds['indices'])
            indptr = np.array(ds['indptr'])
            matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
            self.matrix = matrix
            self.genes = genes
            self.cells = barcodes
            self.alts = alts
            self.Ngenes = len(genes)
            self.Ncells = len(barcodes)
        self.makeIndices()
        if self.verbose:
            stampTime('Gene expression matrix for ' + str(self.Ncells) + ' cells and ' + str(self.Ngenes) + ' genes read into memory.')
        return
    
    def throwGenesAway(self, gl):
        keppos = []
        for i in range(0, self.Ngenes):
            if self.genes[i] in gl:
                keppos.append(i)
            if self.alts[i] in gl:
                keppos.append(i)
        keppos = list(set(keppos))
        keppos.sort()
        self.matrix = self.matrix[keppos,:]
        self.genes = list(self.genes[_] for _ in keppos)
        self.alts = list(self.alts[_] for _ in keppos)
        self.Ngenes = len(self.genes)
        if self.verbose:
            stampTime('Gene expression matrix filtered against genes list of '+str(self.Ngenes)+' genes.')
        return

    
    def saveMatrix(self, fn, matrix, genome, cells, names, genes):
        with h5py.File(fn, 'w') as f:
            dgroup = f.create_group(genome)
            dgroup.create_dataset('data',  data=matrix.data)
            dgroup.create_dataset('indices',  data=matrix.indices)
            dgroup.create_dataset('indptr',  data=matrix.indptr)
            dgroup.create_dataset('shape',  data=matrix.shape)
            dgroup.create_dataset('barcodes',  data=cells)
            fgroup = f.create_group('features')
            fgroup.create_dataset('id',  data=genes)
            fgroup.create_dataset('name',  data=names)
        if self.verbose:
            stampTime('Filtered gene expression matrix saved.')
        return
    
    def saveSmall(self, fn, ngene, ncell=None):
        matrix = self.matrix[range(0, ngene),:]
        if ncell != None:
            matrix = matrix[:,range(0, ncell)]
            cells = self.cells[:ncell]
        else:
            cells = self.cells
        genes = self.genes[:ngene]
        alts = self.alts[:ngene]
        self.saveMatrix(fn, matrix, self.genome, cells, genes, alts)
        return
    
    def renameCells(self, prefix, barcodehistory):
        cells = []
        for cell in self.cells:
            n = prefix+cell
            if n not in barcodehistory:
                barcodehistory[n] = []
            barcodehistory[n].append(cell)
            cells.append(n)
        self.cells = cells
        if self.verbose:
            stampTime('Prefix "'+prefix+'" added to barcodes.')
        return barcodehistory
    
    def makeIndices(self):
        self.genInd, self.cellInd = {}, {}
        for i in range(0, self.Ngenes):
            self.genInd[self.genes[i]] = i
        for i in range(0, self.Ncells):
            self.cellInd[self.cells[i]] = i
        if self.verbose:
            stampTime('Matrix indices linked to names.')
        return

    def filterMatrix(self, mincell=5, minread=0):
        keepgenes = []
        for i in range(0, self.Ngenes):
            numBigEnough = 0
            matrow = self.matrix.getrow(i)
            for j in range(0, self.Ncells):
                if matrow.getcol(j).mean() > minread:
                    numBigEnough += 1
            if numBigEnough >= mincell:
                keepgenes.append(i)
        self.matrix = self.matrix[keepgenes,:]
        self.genes = list(self.genes[_] for _ in keepgenes)
        self.alts = list(self.alts[_] for _ in keepgenes)
        self.Ngenes = len(self.genes)
        if self.verbose:
            stampTime('Gene expression matrix filtered for a minimum of more than '+str(minread)+' reads in '+str(mincell)+' cells.')
        return

    def rankGenes(self, rankedgenes=100, minread=0):
        self.genMax, self.rankInCell = {}, {}
        for i in range(0, self.Ngenes):
            self.genMax[self.genes[i]] = self.matrix.getrow(i).max()
        for i in range(0, self.Ncells):
            cell = self.cells[i]
            self.rankInCell[cell] = {}
            genlist = self.matrix.getcol(i).toarray()
            genlist = zip(range(0, self.Ngenes), genlist)
            genlist.sort(key=lambda x: x[1], reverse=True)
            if rankedgenes < len(genlist):
                genlist = genlist[:rankedgenes]
            else:
                rankedgenes = len(genlist)
            rank, prevread, c, still_good = 0, 0, 0, True
            while c < rankedgenes and still_good:
                a, b = genlist[c]
                if b != prevread:
                    rank +=1
                if b > minread:
                    self.rankInCell[cell][self.genes[a]] = rank
                else:
                    still_good = False
                c += 1
                prevread = b
        if self.verbose:
            stampTime('Genes ranked based on expression within a cell (top '+str(rankedgenes)+') with more than '+str(minread)+' reads.')
        return
    
    def saveRanks(self, fn):
        o = '\t'.join(['', 'MaxRead'] + list(self.cells))+'\n'
        for gene in self.genes:
            l = []
            for cell in self.cells:
                generank = ''
                if gene in self.rankInCell[cell]:
                    generank = str(self.rankInCell[cell][gene])
                l.append(generank) 
            o += gene + '\t' + str(self.genMax[gene]) + '\t' + '\t'.join(l) + '\n'
        f = open(fn, 'w')
        f.write(o)
        f.close()
        if self.verbose:
            stampTime('Gene ranks saved to file '+fn)
        return
    
    def readRanks(self, fn):
        self.genMax, self.rankInCell = {}, {}
        with open(fn) as f:
            firstline = True
            for line in f:
                line = line.split('\n')[0]
                line = line.split('\r')[0]
                if firstline:
                    firstline = False
                    header = line.split('\t')
                    N = len(header)
                    for cell in header[2:]:
                        self.rankInCell[cell] = {}
                else:
                    dat = line.split('\t')
                    gene = dat[0]
                    self.genMax[gene] = dat[1]
                    for i in range(2, N):
                        v = dat[i]
                        if v != '':
                            try:
                                v = float(v)
                                right_val = True
                            except:
                                right_val = False
                            if right_val:
                                self.rankInCell[self.cells[i-2]][gene] = dat[i]
        if self.verbose:
            stampTime('Genes ranks read from file '+fn)
        return
 
    def __getitem__(self, key):
        if key in self.cellInd:
            i = self.cellInd[key]
            return self.cell(self.matrix.getcol(i), self.genInd)
        else:
            return self.cell(sparse.csc_matrix((self.Ngenes, 1)), self.genInd)
    
    def __str__(self):
        return str(self.Ngenes) + ' genes for ' + str(self.Ncells) + ' cells'
    
    class cell:
        def __init__(self, genes, indexes):
            self.genes = genes
            self.indexes = indexes
            return
        
        def __getitem__(self, key):
            if key in self.indexes:
                i = self.indexes[key]
                return self.genes.getrow(i).mean()
            else:
                return 0

if __name__ == '__main__':
    __main__()