#!/usr/bin/python
# -*- coding: utf-8 -*-

#Usage example: python summarizeChainTable.py /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/mappedClonotypes.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/mappedClonotypes.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/Summa/clonotypeTable.tsv

import sys

def __main__():
    args = sys.argv
    basetable, divtab, kidtab, maptab, outF = args[1:]
    M = {}
    with open(maptab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            M[line[0]] = line[1]
    D1 = {}
    h1 = ['barcode', 'is_cell', 'contig_id', 'high_confidence', 'length', 'chain', 'c_gene', 'full_length', 'productive', 'reads', 'umis', 'nseq', 'aseq', 'ncdr', 'acdr', 'cdrlen', 'addednuc', 'sample', 'segments']
    with open(basetable) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            k = line[0]
            if k in M:
                k = M[k]
                D1[k] = line[1:]
    l1 = ['' for x in line[1:]]
    #oops... diversity is by chain pairs... add to cells instead?

    D2 = {}
    with open(divtab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            D2[line[0]] = line[1:]
    l2 = ['' for x in line]



    D1 = {}
    firstline = True
    with open(basetable) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            if firstline:
                firstline = False
                h1 = line[1:]
            else:
                D1[line[0]] = line[1:]
    l1 = ['' for x in line[1:]]
    D1 = {}
    firstline = True
    with open(basetable) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            if firstline:
                firstline = False
                h1 = line[1:]
            else:
                D1[line[0]] = line[1:]
    l1 = ['' for x in line]
    h2 = ['ConsensusClonotype', 'FullChainSig', 'FullChainMain', 'A1ChainSig', 'B1ChainSig', 'A2ChainSig', 'B2ChainSig', 'A1ChainMain', 'B1ChainMain', 'A2ChainMain', 'B2ChainMain', 'A1B1Sig', 'A2B2Sig', 'A1A2Sig', 'B1B2Sig', 'A1B1Main', 'A2B2Main', 'A1A2Main', 'B1B2Main']
    #h2 = ['ConsensusClonotype', 'FullChainSig', 'FullChainMain', 'A1ChainSig', 'B1ChainSig', 'A2ChainSig', 'B2ChainSig', 'A1ChainMain', 'B1ChainMain', 'A2ChainMain', 'B2ChainMain', 'A1B1Sig', 'A2B2Sig', 'A1A2Sig', 'B1B2Sig', 'A1B1Main', 'A2B2Main', 'A1B2Main', 'A2B1Main', 'A1A2Main', 'B1B2Main']
    o = '\t'.join(h1+h2)+'\n'
    lines = set(D1.keys())|set(D2.keys())
    for line in lines:
        if line in D1:
            r1 = D1[line]
        else:
            r1 = l1[:]
        if line in D2:
            r2 = D2[line]
        else:
            r2 = l2[:]
        o += '\t'.join([line]+r1+r2)+'\n'
    f = open(outF, 'w')
    f.write(o)
    f.close()
    return

if __name__ == '__main__':
    __main__()