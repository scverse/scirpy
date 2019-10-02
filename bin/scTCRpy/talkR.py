#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess

from utils import stampTime, tabWriter, tabReader, readORrun, objectContainer, path_to_scripts

def __main__():
    print 'This module is not intended for direct command line usage. Currently supports import to Python only.'
    moduleTest()
    return

def moduleTest():
    return

def featRchain(experiment):
    forced = experiment.force_recompute
    verbose = experiment.verbose
    fbase = experiment.seqDivFile

    def relax(fn, clonTab=None, verbose= False):
        return

    def exportChainFeatures(fn, clonTab, verbose= False):
        clonTab = experiment.cellTCR.clonotypeTable
        headrow = ['Subtype', 'Clonotype', 'Count', 'nTRA', 'nTRB', 'aTRA', 'aTRB']
        chains = {}
        for clone, info in clonTab.iteritems():
            chains[clone] = [clone, str(info['cellNum']), info['chains']['TRA']['primary_cdr3_nt'], info['chains']['TRB']['primary_cdr3_nt'], info['chains']['TRA']['primary_cdr3_aa'], info['chains']['TRB']['primary_cdr3_aa']]
        tabWriter(chains, fn, colnames=headrow, tabformatter='dict_of_list', rowname=None, addheader=True)
        if verbose:
            stampTime('R script input file compiled to ' + fn)
        return

    def runDiver(fn, inPut=None, verbose= False):
        subprocess.call(['Rscript',  path_to_scripts+'chainDiversityCalc.r', inPut, fn])
        diversities = readDiver(fn)
        if verbose:
            stampTime('R script for chain diversity calculation run, results saved to ' + fn)
        return diversities

    def readDiver(fn, inPut=None, verbose= False):
        #the order is: nucA, nucB, nucC, aaA, aaB, aaC
        def lineParse(data, line, lineNum, firstline, includend, test_only=False):
            if test_only:
                return locals()
            if firstline:
                firstline = False
            else:
                data[line[0]] = line[2:]
            return data, firstline, includend
        diversities = tabReader(fn, tabformatter=lambda x: {}, lineformatter=lineParse)
        if verbose:
            stampTime('Read chain diversities from ' + fn)
        return diversities

    def runKider(fn, inPut=None, verbose= False):
        subprocess.call(['Rscript', path_to_scripts+'kideraCalc.r', inPut, fn])
        kideras = readKider(fn)
        if verbose:
            stampTime('R script for chain diversity calculation run, results saved to ' + fn)
        return kideras

    def readKider(fn, inPut=None, verbose= False):
        def lineParse(data, line, lineNum, firstline, includend, test_only=False):
            if test_only:
                return locals()
            if firstline:
                firstline = False
            else:
                d = {}
                d['A'] = line[2:12]
                d['B'] = line[12:22]
                d['C'] = line[22:32]
                data[line[0]] = d 
            return data, firstline, includend
        kideras = tabReader(fn, tabformatter=lambda x: {}, lineformatter=lineParse)
        if verbose:
            stampTime('Read Kidera factors from ' + fn)
        return kideras

    def addToCells(diversities, kideras):
        for cell, dat in experiment.cellTCR.cellTable.iteritems():
            clonotype = dat['clonotype'] 
            if clonotype in diversities:
                div = diversities[clonotype]
            else:
                div = ['NA', 'NA','NA', 'NA','NA', 'NA']
            dat['diver'] = div
            if clonotype in kideras:
                kid = kideras[clonotype]
            else:
                kid = {'A': ['NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA'], 'B': ['NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA'], 'C': ['NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA','NA', 'NA']}
            dat['kideras'] = kid
        return
    readORrun(fbase+'_r_input.tsv', forced, relax, exportChainFeatures, None, kwargs={'clonTab': experiment.cellTCR.clonotypeTable, 'verbose': verbose})
    diversities = readORrun(fbase+'_r_out.tsv', forced, readDiver, runDiver, {}, kwargs={'inPut': fbase+'_r_input.tsv', 'verbose': verbose})
    kideras = readORrun(experiment.kiderFile, forced, readKider, runKider, {}, kwargs={'inPut': fbase+'_r_input.tsv', 'verbose': verbose})
    addToCells(diversities, kideras)
    return

def exportCellFeatures(x):
    dat = []
    dat.append([
        'cell_barcode',
        'clonotypeID',
        'clontype_alias',
        'clontype_size',
        'primary_alpha_nt',
        'primary_beta_nt',
        'primary_alpha_aa',
        'primary_beta_aa',
        'secondary_alpha_nt',
        'secondary_beta_nt',
        'secondary_alpha_aa',
        'secondary_beta_aa',
        'primary_alpha_expression',
        'primary_beta_expression',
        'secondary_alpha_expression',
        'secondary_beta_expression',
        'primary_alpha_nt_diversity',
        'primary_beta_nt_diversity',
        'combined_nt_diversity',
        'primary_alpha_aa_diversity',
        'primary_beta_aa_diversity',
        'combined_aa_diversity',
        'KideraA1',
        'KideraA2',
        'KideraA3',
        'KideraA4',
        'KideraA5',
        'KideraA6',
        'KideraA7',
        'KideraA8',
        'KideraA9',
        'KideraA10',
        'KideraB1',
        'KideraB2',
        'KideraB3',
        'KideraB4',
        'KideraB5',
        'KideraB6',
        'KideraB7',
        'KideraB8',
        'KideraB9',
        'KideraB10',
        'KideraC1',
        'KideraC2',
        'KideraC3',
        'KideraC4',
        'KideraC5',
        'KideraC6',
        'KideraC7',
        'KideraC8',
        'KideraC9',
        'KideraC10'
    ])
    cells, clonotypes = x.cellTCR.cellTable, x.cellTCR.clonotypeTable
    fn = x.cellFeatFile
    for cell, info in cells.iteritems():
        row = [cell]
        clonotype = clonotypes[info['clonotype']]
        row.append(info['clonotype'])
        row.append(clonotype['alias'])
        row.append(clonotype['cellNum'])
        row += [
            clonotype['chains']['TRA']['primary_cdr3_nt'],
            clonotype['chains']['TRB']['primary_cdr3_nt'],
            clonotype['chains']['TRA']['primary_cdr3_aa'],
            clonotype['chains']['TRB']['primary_cdr3_aa'],
            clonotype['chains']['TRA']['secondary_cdr3_nt'],
            clonotype['chains']['TRB']['secondary_cdr3_nt'],
            clonotype['chains']['TRA']['secondary_cdr3_aa'],
            clonotype['chains']['TRB']['secondary_cdr3_aa'],
            str(info['chain_epression']['TRA']),
            str(info['chain_epression']['TRB']),
            str(info['chain_epression']['_TRA']),
            str(info['chain_epression']['_TRB'])
        ]
        row += info['diver']
        row += info['kideras']['A']
        row += info['kideras']['B']
        row += info['kideras']['C']
        clonotype['chains']['TRA']['primary_cdr3_nt']
        dat.append(row)
    tabWriter(dat, fn, tabformatter='list_of_list')
    return

def groupDiversity(x):
    verbose = x.verbose
    forced = x.force_recompute
    fbase = x.clonDivFile
    rFile = fbase+'_r_out.tsv'
    inPut = fbase+'_r_input.tsv'
    ct_dic = {}
    def readDiv(fn, x=x, inPut=inPut, ct_dic=ct_dic, verbose=True):
        def lineParse(data, line, lineNum, firstline, includend, test_only=False):
            if test_only:
                return locals()
            if firstline:
                firstline = False
            else:
                data[line[0]] = float(line[1])
            return data, firstline, includend
        diversities= tabReader(fn, tabformatter=lambda x: {}, lineformatter=lineParse)
        ct_dic = tabReader(inPut+'_recode.txt', tabformat='dict_of_list', evenflatter=True)
        return diversities, ct_dic
    def runDiv(fn, x=x, inPut=inPut, ct_dic=ct_dic, verbose=True):
        clonotype_freqs, clonotype_cells, header = {}, {}, []
        ct_n = 0
        for k, v in x.cellTCR.clonotypeTable.iteritems():
            clonotype_freqs[k] = []
            clonotype_cells[k] = set(v['cells'])
        for respect, group in x.cellGroups.iteritems():
            if respect not in ['major clonotypes']:
                for k, v in group.iteritems():
                    v = set(v)
                    ct_n += 1
                    cat = 'Cat'+str(ct_n)
                    ct_dic[cat] = k
                    header.append(respect.replace(' ', '_')+'.'+cat)
                    for c, d in clonotype_freqs.iteritems():
                        d.append(len(v&clonotype_cells[c]))
            if respect not in ['samples', 'major clonotypes']:
                for sk, sv in x.cellGroups['samples'].iteritems():
                    sv = set(sv)
                    srespect =  sk + 'xXxXxXxXxXxXxXxXx' + respect
                    for k, v in group.iteritems():
                        ct_n += 1
                        cat = 'Cat'+str(ct_n)
                        ct_dic[cat] = k
                        v = set(v)&sv
                        header.append(srespect.replace(' ', '_')+'.'+cat)
                        for c, d in clonotype_freqs.iteritems():
                            d.append(len(v&clonotype_cells[c]))
        tabWriter(clonotype_freqs, inPut, tabformatter='dict_of_list', colnames=header, rowname='CLONOTYPE')
        subprocess.call(['Rscript', path_to_scripts+'cloneDiversityCalc.r', inPut, fn])
        if verbose:
            stampTime('Clonotype diversity calculations saved to ' + fn)
        tabWriter(ct_dic, inPut+'_recode.txt', tabformatter='dict_of_singles')
        diversities, nothing = readDiv(fn)
        return diversities, ct_dic
    diversities, ct_dic = readORrun(rFile, forced, readDiv, runDiv, [{},{}], kwargs={'x': x, 'inPut': inPut, 'ct_dic': ct_dic, 'verbose': verbose})
    groupdiv = {'SAMPLESPLIT': {}}
    for k, v in diversities.iteritems():
        sg = groupdiv
        if k.find('xXxXxXxXxXxXxXxXx') > 0:
            s, k = k.split('xXxXxXxXxXxXxXxXx')
            if s not in groupdiv['SAMPLESPLIT']:
                groupdiv['SAMPLESPLIT'][s] = {}
            sg = groupdiv['SAMPLESPLIT'][s]
        k = k.replace('_', ' ')
        a, b = k.split('.', 1)
        b = ct_dic[b]
        if a not in sg:
            sg[a] = {}
        sg[a][b] = v
    x.divGroup=groupdiv
    return

if __name__ == '__main__':
    __main__()