#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

from parseTCR import scTCR, defaultCellGroups
from parseGex import allGex, gexMat
from clusterTools import manyMani
from distanceMetrics import distList
from utils import stampTime, objectContainer

def __main__():
    print 'This module is not intended for direct command line usage. Currently supports import to Python only.'
    moduleTest()
    return

def moduleTest():
    experiment = eXperiment(projectRoot='testdata', verbose=True)
    experiment.addSample('pbmc', 'Control', 'S1_', t_tcrDir='', t_gexDir='')
    experiment.addSample('nsclc', 'Cancer', 'S2_', t_tcrDir='', t_gexDir='')
    experiment.addProjection('basic', gex=experiment.cellGex.gex, coordFile='tsne_coord.tsv', annotFile='tsne_annot.tsv', force_recompute=True)
    return
    
class eXperiment:
    def __init__(self, projectRoot=None, outDir=None, reportFile='report.html', kiderFile='kidera_factors.tsv', cellFeatFile='tcr_computed_cell_features.tsv', cellGenFile='gene_normalization_metadata', genListFile='genelist.txt', seqDivFile='sequence_based_diversity', clonDivFile='clonotype_based_diversity', distanceBase='distance', gliphBase='gliph', barcodeRe={}, cellTCR=None, cellGex=None, maniFolds=None, xSamples=None, cellDistances=None, cellGroups=None, criticalClonotypeSize=None, numCores=8, force_recompute=False, verbose=False):
        self.numCores=numCores
        self.force_recompute=force_recompute
        self.verbose=verbose
        self.barcodeRe=barcodeRe
        if cellGroups == None:
            self.cellGroups = defaultCellGroups
        if projectRoot == None:
            self.projectRoot = os.path.dirname(os.getcwd()) + '/'
        else:
            self.projectRoot = projectRoot + '/'
        if projectRoot == '/':
            projectRoot = ''
        if projectRoot == '//':
            projectRoot = '/'
        if outDir == None:
            self.outDir = self.projectRoot + 'Reports/'
        else:
            self.outDir = self.projectRoot + outDir + '/'
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.reportFile = self.outDir + reportFile
        self.kiderFile = self.outDir + kiderFile
        self.cellFeatFile = self.outDir + cellFeatFile
        self.cellGenFile = self.outDir + cellGenFile
        self.genListFile = self.projectRoot + genListFile
        self.seqDivFile = self.outDir + seqDivFile
        self.clonDivFile = self.outDir + clonDivFile
        self.distanceBase = self.outDir + distanceBase
        self.gliphBase = self.outDir + gliphBase
        if cellTCR == None:
            self.cellTCR = scTCR(verbose=self.verbose, criticalClonotypeSize=criticalClonotypeSize)
        else:
            self.cellTCR = cellTCR
        if cellGex == None:
            self.cellGex = allGex()
        else:
            self.cellGex = cellGex
        if maniFolds == None:
            self.maniFolds = manyMani()
        else:
            self.maniFolds = maniFolds
        if cellDistances == None:
            self.cellDistances = distList()
        else:
            self.cellDistances = cellDistances
        if xSamples == None:
            self.xSamples = self.sampleList()
        else:
            self.xSamples = xSamples
        self.sampleOrder = []
        return
    
    def __str__(self):
        return 'Experimental layout containing ' + str(len(self.xSamples)) + 'items'
    
    def __getitem__(self, key):
        return self.xSamples[key]

    class sampleList(objectContainer):
        def add(self, sample, force_override=False):
            if hasattr(sample, 'label'):
                if sample.label not in self.data:
                    self.order.append(sample.label)
                    self.data[sample.label] = sample
                else:
                    if force_override:
                        self.order.append(sample.label)
                        self.data[sample.label] = sample
                        print 'Duplicate sample labels! Previous data has been overwritten.'
                    else:
                        print 'Duplicate sample labels! Nothing changed.'
            else:
                print 'The sample has to have a label in order to add it to a container.'
            return

    class newSample:
        def __init__(self, sprefix, sname, slabel, projectDir, gexNorm, tcrDir='TCR', gexDir='Genes', tcr_filtered=True, gex_filtered=True):
            self.name = sname
            self.label = slabel
            if tcrDir == '':
                self.tcrDir = projectDir
            else:
                self.tcrDir = projectDir + tcrDir + '/'
            if gexDir == '':
                self.gexDir = projectDir
            else:
                self.gexDir= projectDir + gexDir + '/'
            #self.contigFile = self.tcrDir + sprefix + '_t_all_contig_annotations.json'
            if tcr_filtered:
                self.contigTabFile = self.tcrDir + sprefix + 'filtered_contig_annotations.csv'
            else:
                self.contigTabFile = self.tcrDir + sprefix + 'all_contig_annotations.csv'
            self.consensusFile = self.tcrDir + sprefix + 'consensus_annotations.json'
            if gex_filtered:
                #self.gexFile = self.gexDir + sprefix + 'filtered_gene_bc_matrices_h5.h5'
                self.gexFile = self.gexDir + sprefix + 'filtered_feature_bc_matrix.h5'
            else:
                #self.gexFile = self.gexDir + sprefix + 'raw_gene_bc_matrices_h5.h5'
                self.gexFile = self.gexDir + sprefix + 'raw_feature_bc_matrix.h5'
            self.gexnormFile = gexNorm
            return
        
        def __str__(self):
            return self.name
    
    def addSample(self, sprefix, sname, slabel, **kwargs):
        if self.verbose:
            stampTime('Adding sample ' + sname + '...')
        a = locals()
        a = a['kwargs']
        tcr_args = {}
        gex_args = {'sample_prefix': slabel, 'ranksfn': self.cellGenFile+'_'+sname+'.tsv', 'keeponlygenes': self.genListFile}
        if 'force_recompute' not in gex_args:
            gex_args['force_recompute'] = self.force_recompute
        if 'force_recompute' not in gex_args:
            gex_args['verbose'] = self.verbose
        if 'barcodehistory' not in gex_args:
            gex_args['barcodehistory'] = self.barcodeRe
        for k, v in a.iteritems():
            if k[:2] =='t_':
                tcr_args[k[2:]] = v
            if k[:2] =='g_':
                gex_args[k[2:]] = v
        sample = self.newSample(sprefix, sname, slabel, self.projectRoot, self.cellGenFile, **tcr_args)
        self.xSamples.add(sample)
        self.sampleOrder.append(sname)
        self.cellTCR.addSampleTCR(sample.contigTabFile, sample.consensusFile, slabel, sname, barcodeRe=self.barcodeRe)
        smat = gexMat(sample.gexFile, **gex_args)
        self.cellGex.add(smat)
        self.barcodeRe = self.cellGex.barcodehistory
        if self.verbose:
            stampTime('A new sample called ' + sname + ' added.')
        return
    
    def distanceCalc(self, name, **kwargs):
        if self.verbose:
            stampTime('Initializing distance matrix ' + name + '...')
        a = locals()
        a = a['kwargs']
        if 'matrixFile' not in a:
            a['matrixFile'] = self.distanceBase
            a['projectRoot'] = ''
        else:
            if 'projectRoot' not in a:
                a['projectRoot'] = self.projectRoot
        if 'force_recompute' not in a:
            a['force_recompute'] = self.force_recompute
        if 'verbose' not in a:
            a['verbose'] = self.verbose
        self.cellDistances.add(name, self.cellTCR.cellTable, self.cellTCR.clonotypeTable, **a)
        self.cellGroups['clusters in '+name] = self.cellDistances[name].treeGroups
        if self.verbose:
            stampTime('A new distance matrix called ' + name + ' added.')
        return
    
    def addProjection(self, name, **kwargs):
        if self.verbose:
            stampTime('Adding ' + name + ' mainfold...')
        a = locals()
        if 'verbose' not in a:
            a['verbose'] = self.verbose
        if 'force_recompute' not in a:
            a['force_recompute'] = self.force_recompute
        a = a['kwargs']
        if 'projectRoot' not in a:
            a['projectRoot'] = self.projectRoot
        if 'numCores' not in a:
            a['numCores'] = self.numCores
        if 'force_recompute' not in a:
            a['force_recompute'] = self.force_recompute
        if 'verbose' not in a:
            a['verbose'] = self.verbose
        self.maniFolds.add(name, **a)
        self.cellGroups['clusters in '+name] = self.maniFolds[name].cellTypes
        if self.verbose:
            stampTime('Projection of the ' + name + ' manifold added.')
        return

if __name__ == '__main__':
    __main__()