#!/usr/bin/python
# -*- coding: utf-8 -*-

import json

from utils import stampTime, tabReader, namedMatrix

def __main__():
    print 'This module is not intended for direct command line usage. Currently supports import to Python only.'
    #A table maker function could be added in the future so that parse data could be saved and passed to R
    moduleTest()
    return

def moduleTest():
    print 'Testing module by parsing test TCR data'
    prefix, name = 'S1_', 'Control'
    consensusFile = 'testdata/pbmc_t_consensus_annotations.json'
    contigTabFile = 'testdata/pbmc_t_filtered_contig_annotations.csv'
    tcr = scTCR(verbose=True)
    tcr.addSampleTCR(contigTabFile, consensusFile, prefix, name)
    print 'Number of cells:', len(tcr.cellTable)
    print 'Cell keys:', tcr.cellTable.keys()[:5]
    print 'A cell example:'
    print tcr.cellTable[tcr.cellTable.keys()[0]]
    print 'Consensus keys:', tcr.consensusTable.keys()[:5]
    print 'A consensus example:'
    print tcr.consensusTable[tcr.consensusTable.keys()[0]]
    print 'Clonotype keys:', tcr.clonotypeTable.keys()[:5]
    print 'A clonotype example:'
    print tcr.clonotypeTable[tcr.clonotypeTable.keys()[0]]
    return

defaultCellGroups = {
    'samples': {},
    'chain pairing': {
        'Single pair': [],
        'Double alpha': [],
        'Double beta': [],
        'Double orphan alpha': [],
        'Double orphan beta': [],
        'Orphan alpha': [],
        'Orphan beta': [],
        'Full doublet': []
        },
    'TRA V segment usage': {},
    'TRA D segment usage': {},
    'TRA J segment usage': {},
    'TRB V segment usage': {},
    'TRB D segment usage': {},
    'TRB J segment usage': {},
    'major clonotypes': {}
    }

class scTCR:
    def __init__(self, contigTable=None, barcodeRe={}, cellTable={}, consensusTable={}, clonotypeTable={}, clonaliasTable=[{}, {}], cdrTable={}, cellGroups=None, criticalClonotypeSize=4, verbose=False):
        self.barcodeRe=barcodeRe
        self.cellTable=cellTable
        if contigTable == None:
            self.contigTable = namedMatrix(unique=False)
        else:
            self.contigTable=contigTable
        if cellGroups == None:
            self.cellGroups = defaultCellGroups.copy()
        else:
            self.contigTable=contigTable
        self.consensusTable=consensusTable
        self.clonotypeTable=clonotypeTable
        self.clonaliasTable=clonaliasTable
        self.cdrTable=cdrTable
        self.cellGroups=cellGroups
        self.criticalClonotypeSize=criticalClonotypeSize
        self.verbose=verbose
        return
    
    def addSampleTCR(self, contigTabFile, consensusFile, prefix, sname, barcodeRe=None):
        if barcodeRe != None:
            self.barcodeRe = barcodeRe
        self.contigTable, self.barcodeRe, self.cellTable, self.clonotypeTable, self.clonaliasTable, self.cdrTable, self.cellGroups = self.clonotypeFromContigs(contigTabFile, prefix, sname, contigTable=self.contigTable, barcodeRe=self.barcodeRe, cellTable=self.cellTable, clonotypeTable=self.clonotypeTable, clonaliasTable=self.clonaliasTable, cdrTable=self.cdrTable, cellGroups=self.cellGroups)
        self.consensusTable, self.parseConsensus(consensusFile, prefix, self.consensusTable)
        if self.verbose:
            stampTime('TCR data of a new sample with prefix ' + prefix + ' added.')
        return

    def clonotypeFromContigs(self, fn, prefix, sname, contigTable=None, barcodeRe={}, cellTable={}, clonotypeTable={}, clonaliasTable=[{}, {}], cdrTable={}, cellGroups=None):
        if cellGroups == None:
            cellGroups = defaultCellGroups.copy()

        def lineParse(data, line, lineNum, firstline, includend, test_only=False):
            if test_only:
                return locals()
            if firstline:
                firstline = False
                header = line[1:]
                data.setColNames(line[1:])
            else:
                key = prefix+line[0]
                line = line[1:]
                readcol = data.colnames['reads']
                reads = line[readcol]
                try:
                    reads = int(reads)
                except:
                    pass
                line[readcol] = reads
                cdr3 = line[data.colnames['cdr3']]
                if cdr3 == 'None':
                    cdr3 = ''
                cdr3 = cdr3.replace('*', 'X')
                line[data.colnames['cdr3']] = cdr3
                cdr3_nt = line[data.colnames['cdr3_nt']]
                if cdr3_nt == 'None':
                    cdr3_nt = ''
                line[data.colnames['cdr3_nt']] = cdr3_nt
                dat = data.matrixRow(key, line, data)
                data.add(dat)
            return data, firstline, includend

        contigTable = tabReader(fn, data=contigTable, tabformatter=lambda x: x, lineformatter=lineParse)
        bySample = cellGroups['samples']
        bySample[sname] = []
        bySample = bySample[sname]
        byChain = cellGroups['chain pairing']
        bySegment = {
            'TRAV': cellGroups['TRA V segment usage'],
            'TRAD': cellGroups['TRA D segment usage'],
            'TRAJ': cellGroups['TRA J segment usage'],
            'TRBV': cellGroups['TRB V segment usage'],
            'TRBD': cellGroups['TRB D segment usage'],
            'TRBJ': cellGroups['TRB J segment usage']
        }
        n = 0
        for m, cell, rowdata in contigTable:
            bc = cell.split(prefix)
            if len(bc) == 2:
                bc = bc[1]
                if cell not in barcodeRe:
                    barcodeRe[cell] = []
                barcodeRe[cell].append(bc)
                bySample.append(cell)
                for g in ['v_gene', 'd_gene', 'j_gene', 'c_gene']:
                    e = rowdata[g]
                    for s in e:
                        if s[:4] in bySegment:
                            seg = bySegment[s[:4]]
                            if s not in seg:
                                seg[s] = []
                            bySegment[s[:4]][s].append(cell)
                numcellchains, cellchains = len(e), {'TRA': ['', 0, '', ''], 'TRB': ['', 0, '', ''], '_TRA': ['', 0, '', ''], '_TRB': ['', 0, '', '']}
                n += numcellchains
                for i in range(0, numcellchains):
                    cdr3 = rowdata['cdr3'][i]
                    cdr3_nt = rowdata['cdr3_nt'][i]
                    chain = rowdata['chain'][i]
                    reads = rowdata['reads'][i]
                    cons = prefix+rowdata['raw_consensus_id'][i]
                    if chain in ['TRA', 'TRB']:
                        if cellchains[chain][0] == '':
                            cellchains[chain] = [cdr3, reads, cdr3_nt, cons]
                        else:
                            if reads > cellchains[chain][1] :
                                if cdr3 == '':
                                    if cellchains['_'+chain] == '' and reads > cellchains['_'+chain][1]:
                                        cellchains['_'+chain] = [cdr3, reads, cdr3_nt, cons]
                                else:
                                    cellchains['_'+chain] = cellchains[chain][:]
                                    cellchains[chain] = [cdr3, reads, cdr3_nt, cons]
                            else:
                                cellchains['_'+chain] = [cdr3, reads, cdr3_nt, cons]
                if cellchains['TRA'][0] == '':
                    if cellchains['TRB'][0] == '':
                        pass
                    else:
                        if cellchains['_TRB'][0] == '':
                            byChain['Orphan beta'].append(cell)
                        else:
                            byChain['Double orphan beta'].append(cell)
                else:
                    if cellchains['TRB'][0] == '':
                        if cellchains['_TRA'][0] == '':
                            byChain['Orphan alpha'].append(cell)
                        else:
                            byChain['Double orphan alpha'].append(cell)
                    else:
                        if cellchains['_TRB'][0] == '':
                            if cellchains['_TRA'][0] == '':
                                byChain['Single pair'].append(cell)
                            else:
                                byChain['Double alpha'].append(cell)
                        else:
                            if cellchains['_TRA'][0] == '':
                                byChain['Double beta'].append(cell)
                            else:
                                byChain['Full doublet'].append(cell)
                cloneID = '|'.join([cellchains[x][0] for x in ['TRA', 'TRB', '_TRA', '_TRB']])
                if cloneID in clonaliasTable[0]:
                    clonotype = clonaliasTable[0][cloneID]
                else:
                    clonotype = 'ct_'+cell
                    clonaliasTable[0][cloneID] = clonotype
                    clonotypeTable[clonotype] = {
                        'cells': [], 
                        'cdrSym': cloneID, 
                        'alias': '', 
                        'chains': {
                            'TRA': {
                                'primary_cdr3_aa': cellchains['TRA'][0],
                                'primary_cdr3_nt': cellchains['TRA'][2],
                                'primary_consensus': cellchains['TRA'][3],
                                'secondary_cdr3_aa': cellchains['_TRA'][0],
                                'secondary_cdr3_nt': cellchains['_TRA'][2],
                                'secondary_consensus': cellchains['_TRA'][3]
                                },
                            'TRB': {
                                'primary_cdr3_aa': cellchains['TRB'][0],
                                'primary_cdr3_nt': cellchains['TRB'][2],
                                'primary_consensus': cellchains['TRB'][3],
                                'secondary_cdr3_aa': cellchains['_TRB'][0],
                                'secondary_cdr3_nt': cellchains['_TRB'][2],
                                'secondary_consensus': cellchains['_TRB'][3]
                                }
                            }
                        }
                clonotypeTable[clonotype]['cells'].append(cell)
                for i in range(0, numcellchains):
                    cdr = rowdata['cdr3'][i]
                    inalfa, inbeta = 0, 0
                    chain = rowdata['chain'][i]
                    if chain == 'TRA':
                        inalfa += 1
                    if chain == 'TRB':
                        inbeta += 1
                    if cdr not in cdrTable:
                        cdrTable[cdr] = {'nt_seq': [rowdata['cdr3_nt'][i]], 'cells': [cell], 'clonotypes': [clonotype]}
                    else:
                        cdrTable[cdr]['nt_seq'].append(rowdata['cdr3_nt'][i])
                        cdrTable[cdr]['cells'].append(cell)
                        cdrTable[cdr]['clonotypes'].append(clonotype)
                cellTable[cell] = {'clonotype': clonotype, 'chain_epression': {'TRA': cellchains['TRA'][1], 'TRB': cellchains['TRB'][1], '_TRA': cellchains['_TRA'][1], '_TRB': cellchains['_TRB'][1]}}
        numPairTcells = len(set(bySample)&set(cellGroups['chain pairing']['Single pair']))
        if self.verbose:
            stampTime('TCR sequences of ' + str(n) + ' contigs of ' + str(len(bySample)) + ' cells parsed (' + str(numPairTcells) + ' of these have a single TCR chain pair).')
        return contigTable, barcodeRe, cellTable, clonotypeTable, clonaliasTable, cdrTable, cellGroups

    def collectSegments(self, annotations, cdrStart, cdrStop):
        segments, segline = [], []
        for anne in annotations:
            segments.append([anne['feature']['gene_name'], anne['contig_match_start'], anne['contig_match_end']])
        chain = anne['feature']['chain']
        segments.sort(key=lambda x: x[2])
        previ, inserted = 0, 0
        for s in segments:
            difi = s[1] - previ
            if previ >= cdrStart:
                if s[1] <= cdrStop:
                    inserted += difi
            previ = s[2]
            if previ >= cdrStart:
                if s[1] <= cdrStop:
                    segline.append(s[0])
        return inserted, segments, segline, chain

    def parseConsensus(self, fn, prefix, consensusTable):
        with open(fn) as f:
            data = json.load(f)
        for clone in data:
            contid, clonotype, seq, aas, info = prefix+clone['contig_name'], prefix+clone['clonotype'], clone['sequence'], clone['aa_sequence'], clone['info']
            cdrAA, cdrNuc, cdrStart, cdrStop = clone['cdr3'], clone['cdr3_seq'], clone['cdr3_start'], clone['cdr3_stop']
            highConf, mayProd, beenFilt = clone['high_confidence'], clone['productive'], clone['filtered']
            inserted, segments, segline, chain = self.collectSegments(clone['annotations'], cdrStart, cdrStop)
            consensusTable[contid] = {
                'clonotype': clonotype,
                'nseq': seq,
                'aseq': aas,
                'ncdr': cdrNuc,
                'acdr': cdrAA,
                'conf': highConf,
                'prod': mayProd,
                'filt': beenFilt,
                'chain': chain,
                'segments': segline,
                'cdrlen': len(cdrAA),
                'addednuc': inserted
            }
            for k, v in info.iteritems():
                if k not in ['cells', 'cell_contigs']:
                    consensusTable[contid][k] = v
            consensusTable[contid]['cells'] = [prefix+x for x in info['cells']]
            consensusTable[contid]['cell_contigs'] = [prefix+x for x in info['cell_contigs']]
        if self.verbose:
            stampTime('Consensus contig annotations parsed.')
        return consensusTable
    
    def reassessClonotypes(self):
        sortand = []
        for clone, info in self.clonotypeTable.iteritems():
            N = len(info['cells'])
            info['cellNum'] = N
            if N >= self.criticalClonotypeSize:
                if info['cdrSym'] != '|||':
                    sortand.append([clone, N])
        sortand.sort(key= lambda x: x[1], reverse=True)
        n = 0
        for a, b in sortand:
            n += 1
            newname = 'clonotype_'+str(n)
            self.clonotypeTable[a]['alias'] = newname
            self.cellGroups['major clonotypes'][newname] = self.clonotypeTable[a]['cells'][:]
            self.clonaliasTable[1][newname] = a
        if self.verbose:
            stampTime('After reassignment of clonotypes, ' + str(len(self.cellGroups['major clonotypes'])) + ' major clonotypes were found.')
        return

if __name__ == '__main__':
    __main__()