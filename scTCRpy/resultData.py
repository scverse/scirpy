#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import fig2hex
import plotGraphs
import numpy as np

def __main__():
    print 'This module is not intended for direct command line usage. Currently suports import to Python only.'
    return

class clonoTable():
    def __init__(self, experiment):
        samples, clonogroups, tcrdata, order = experiment.cellGroups['samples'], experiment.cellGroups['major clonotypes'], experiment.cellTCR, experiment.sampleOrder 
        table, bar, xticks = [], [], []
        if order == None:
            order = list(samples.keys())
        condlist = []
        for c in order:
            condlist.append('Cells in '+c)
            bar.append([])
        clonodat = []
        for k, v in clonogroups.iteritems():
            clonodat.append([k, len(v), v])
        clonodat.sort(key=lambda x: int(x[0].split('_')[1]))
        clonodat = clonodat[:20]
        mCl = 0
        for ct, size, cells in clonodat:
            xticks.append(ct)
            cells = set(cells)
            tabrow = [ct]
            cdr = tcrdata.clonotypeTable[tcrdata.clonaliasTable[1][ct]]['cdrSym']
            cdr = cdr.split('|')
            tabrow += cdr
            tabrow.append(str(size))
            sasi = []
            for i in range(0, len(order)):
                s = order[i]
                scells = set(samples[s])&cells
                numcell = len(scells)
                if numcell > mCl:
                    mCl = numcell
                sasi.append(numcell)
                bar[i].append(numcell)
            tabrow += [str(x) for x in sasi]
            tabrow.append(plotGraphs.divBar(sasi, mCl))
            table.append(tabrow)
        self.table = table
        self.bar = bar
        self.order = order
        self.xticks = xticks
        self.headers = ['Clonotype', 'Dominant TRA CDR3', 'Dominant TRB CDR3', 'Secondary TRA CDR3', 'Secondary TRB CDR3', 'Number of cells'] + condlist + ['<div style="width: 50px; height: 20px;"></div>']
        self.ax, self.fig = plotGraphs.plotGrBar(self.bar, self.order, xticks=self.xticks, ylabel='Number of cells', figdim={'figsize': (7.2,4.5), 'dpi': 600})
        return

    def html(self, divname='clonotype_table'):
        div = {
            'class': 'toggle_tab',
            'id': divname,
            'children': [
                {
                    'class': 'widefig',
                    'uid': divname,
                    'figure': self.fig,
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                },
                {
                    'eleName': 'table',
                    'id': 'global_clonotype_table',
                    'uid': divname+'_1',
                    'innerHTML': '<tr><th>' + '</th><th>'.join(self.headers) + '</th></tr>\n' + '\n'.join(['<tr><td>' + '</td><td>'.join(row) + '</td></tr>' for row in self.table]) + '\n'
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Clonotype frequency',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class clonOverlap:
    def __init__(self, experiment):
        samples, tcrdata, order = experiment.cellGroups['samples'], experiment.cellTCR, experiment.sampleOrder 
        table = []
        figures = {}
        if order == None:
            order = list(samples.keys())
        selections = {'x_value': order[:], 'y_value': order[:]}
        self.selorder = ['x_value', 'y_value']
        for s1 in order:
            for s2 in order:
                if s1 != s2:
                    overlaps = {}
                    c1 = set(samples[s1])
                    c2 = set(samples[s2])
                    for k, v in tcrdata.clonotypeTable.iteritems():
                        if v['cdrSym'] != '|||':
                            cells = set(v['cells'])
                            l1 = len(cells&c1)
                            l2 = len(cells&c2)
                            lapkey = str(l1)+'_'+str(l2)
                            if lapkey not in ['1_0', '0_1']:
                                if lapkey not in overlaps:
                                    overlaps[lapkey] = 0
                                overlaps[lapkey] += 1
                    x, y, z = [], [], []
                    for k, v in overlaps.iteritems():
                        l1, l2 = k.split('_')
                        x.append(int(l1))
                        y.append(int(l2))
                        z.append(v)
                    table.append([s1, s2, x, y, z])
                    ax, fig = plotGraphs.plotPairBubble(s1, s2, x, y, z)
                    figures[s1+'_'+s2] = fig
        self.table = table
        self.figures = figures
        self.selections = selections
        return
    
    def html(self, divname='clone_overlap'):
        selectors = ''
        for ki in range(0, len(self.selorder)):
            k = self.selorder[ki]
            v = self.selections[k]
            selectors += '<div class="selabel">'+k+'</div><select onchange="update_by_choice(this)">'
            for i in range(0, len(v)):
                e = v[i]
                if i == ki:
                    selectors += '<option selected value="'+e+'">'+e+'</option>'
                else:
                    selectors += '<option value="'+e+'">'+e+'</option>'
            selectors += '</select>'
        div = {
            'id': divname,
            'class': 'toggle_tab',

            'children': [
                {
                    'class': 'chooser',
                    'id': divname+'_chooser',
                    'uid': divname+'_chooser',
                    'innerHTML': selectors
                }
            ]
            }
        for k, v in self.figures.iteritems():
            fd = {
                'class': 'medfig',
                'id': divname+'_'+k,
                'uid': divname+k,
                'figure': v,
                'innerHTML': ''
                }
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Clonotype overlaps\nbetween samples',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class exprTVio:
    def __init__(self, experiment):
        samples, cells, order = experiment.cellGroups['samples'], experiment.cellTCR.cellTable, experiment.sampleOrder
        table = {'Sample': []}
        if order == None:
            order = list(samples.keys())
        collabels = ['Dominant alpha', 'Secondary alpha', 'Dominant beta', 'Secondary beta']
        for l in collabels:
            table[l] = []
        sd = {}
        for k, v in samples.iteritems():
            for c in v:
                sd[c] = k
        for c, d in cells.iteritems():
            chains = d['chain_epression']
            table['Sample'].append(sd[c])
            table['Dominant alpha'].append(chains['TRA'])
            table['Dominant beta'].append(chains['TRB'])
            table['Secondary alpha'].append(chains['_TRA'])
            table['Secondary beta'].append(chains['_TRB'])
        self.table=table
        self.collabels = collabels
        self.order = order
        self.ax, self.fig = plotGraphs.plotDistribution(self.table, 'Expression of TCR chains', 'No of reads', hue='Sample', order=self.collabels, huerder=self.order, show_strip=True)
        return
    
    def html(self, divname='expr_tcr'):
        div = {
            'class': 'toggle_sub',
            'children': [
                {
                    'class': 'smallfig',
                    'uid': divname,
                    'figure': self.fig,
                    'innerHTML': ''
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'tcr_stats',
            }
        return argdic

class lenCdrIns:
    def __init__(self, experiment):
        samples, cells, clonotypes, consenses, order = experiment.cellGroups['samples'], experiment.cellTCR.cellTable, experiment.cellTCR.clonotypeTable, experiment.cellTCR.consensusTable, experiment.sampleOrder
        table = {'Sample': []}
        if order == None:
            order = list(samples.keys())
        collabels = ['Dominant alpha', 'Secondary alpha', 'Dominant beta', 'Secondary beta']
        for l in collabels:
            table[l] = []
        sd = {}
        for k, v in samples.iteritems():
            for c in v:
                sd[c] = k
        for c, d in cells.iteritems():
            chains = clonotypes[d['clonotype']]['chains']
            table['Sample'].append(sd[c])
            da, db, sa, sb = 0, 0, 0, 0
            try:
                da = consenses[chains['TRA']['primary_consensus']]['addednuc']
            except:
                pass
            try:
                db = consenses[chains['TRB']['primary_consensus']]['addednuc']
            except:
                pass
            try:
                sa = consenses[chains['TRA']['secondary_consensus']]['addednuc']
            except:
                pass
            try:
                sb = consenses[chains['TRB']['secondary_consensus']]['addednuc']
            except:
                pass
            table['Dominant alpha'].append(da)
            table['Dominant beta'].append(db)
            table['Secondary alpha'].append(sa)
            table['Secondary beta'].append(sb)
        self.table = table
        self.collabels = collabels
        self.order = order
        self.ax, self.fig = plotGraphs.plotDistribution(self.table, 'Length of CDR3 region', 'Inserted nt', hue='Sample', order=self.collabels, huerder=self.order, show_strip=True)
        return
    
    def html(self, divname='adn_tcr'):
        div = {
            'class': 'toggle_sub',
            'children': [
                {
                    'class': 'smallfig',
                    'uid': divname,
                    'figure': self.fig,
                    'innerHTML': ''
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'tcr_stats',
            }
        return argdic

class lenCdrAa:
    def __init__(self, experiment):
        samples, cells, clonotypes, order = experiment.cellGroups['samples'], experiment.cellTCR.cellTable, experiment.cellTCR.clonotypeTable, experiment.sampleOrder
        table = {'Sample': []}
        if order == None:
            order = list(samples.keys())
        collabels = ['Dominant alpha', 'Secondary alpha', 'Dominant beta', 'Secondary beta']
        for l in collabels:
            table[l] = []
        sd = {}
        for k, v in samples.iteritems():
            for c in v:
                sd[c] = k
        for c, d in cells.iteritems():
            chains = clonotypes[d['clonotype']]['chains']
            table['Sample'].append(sd[c])
            table['Dominant alpha'].append(len(chains['TRA']['primary_cdr3_aa']))
            table['Dominant beta'].append(len(chains['TRB']['primary_cdr3_aa']))
            table['Secondary alpha'].append(len(chains['TRA']['secondary_cdr3_aa']))
            table['Secondary beta'].append(len(chains['TRB']['secondary_cdr3_aa']))
        self.table = table
        self.collabels = collabels
        self.order = order
        self.ax, self.fig = plotGraphs.plotDistribution(self.table, 'Length of CDR3 region', 'Length (aa)', hue='Sample', order=self.collabels, huerder=self.order, show_strip=True)
        return
    
    def html(self, divname='len_tcr'):
        div = {
            'class': 'toggle_sub',
            'children': [
                {
                    'class': 'smallfig',
                    'uid': divname,
                    'figure': self.fig,
                    'innerHTML': ''
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'tcr_stats',
            }
        return argdic

class lenCdrNt:
    def __init__(self, experiment):
        samples, cells, clonotypes, order = experiment.cellGroups['samples'], experiment.cellTCR.cellTable, experiment.cellTCR.clonotypeTable, experiment.sampleOrder
        table = {'Sample': []}
        if order == None:
            order = list(samples.keys())
        collabels = ['Dominant alpha', 'Secondary alpha', 'Dominant beta', 'Secondary beta']
        for l in collabels:
            table[l] = []
        sd = {}
        for k, v in samples.iteritems():
            for c in v:
                sd[c] = k
        for c, d in cells.iteritems():
            chains = clonotypes[d['clonotype']]['chains']
            table['Sample'].append(sd[c])
            table['Dominant alpha'].append(len(chains['TRA']['primary_cdr3_nt']))
            table['Dominant beta'].append(len(chains['TRB']['primary_cdr3_nt']))
            table['Secondary alpha'].append(len(chains['TRA']['secondary_cdr3_nt']))
            table['Secondary beta'].append(len(chains['TRB']['secondary_cdr3_nt']))
        self.table = table
        self.collabels = collabels
        self.order = order
        self.ax, self.fig = plotGraphs.plotDistribution(self.table, 'Length of CDR3 region', 'Length (nt)', hue='Sample', order=self.collabels, huerder=self.order, show_strip=True)
        return
    
    def html(self, divname='lent_tcr'):
        div = {
            'class': 'toggle_sub',
            'children': [
                {
                    'class': 'smallfig',
                    'uid': divname,
                    'figure': self.fig,
                    'innerHTML': ''
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'tcr_stats',
            }
        return argdic

class orphanBar:
    def __init__(self, experiment):
        samples, chains, order = experiment.cellGroups['samples'], experiment.cellGroups['chain pairing'], experiment.sampleOrder
        xticks = ['Single pair', 'Double alpha', 'Double beta', 'Full doublet', 'Orphan alpha', 'Orphan beta', 'Double orphan alpha',  'Double orphan beta']
        table, normtab, rowsums= [], [], []
        if order == None:
            order = list(samples.keys())
        if xticks == None:
            xticks = list(set(chains.keys()))
            xticks.sort()
        for sample in order:
            row, rowsum = [], 0
            cells = set(samples[sample])
            for k in xticks:
                v = chains[k]
                val = len(set(v)&cells)
                row.append(val)
                rowsum += val
            table.append(row)
            rowsums.append(rowsum)
        for i in range(0, len(table)):
            r, s = [], rowsums[i]
            for a in table[i]:
                r.append(float(a)/s)
            normtab.append(r)
        self.order = order
        self.table = table
        self.normtab = normtab
        xticks[6] = '2xO alpha'
        xticks[7] = '2xO beta'
        self.xticks = xticks
        self.barax1, self.barfig1 = plotGraphs.plotHorGrStBar(self.table, self.order, xticks=self.xticks, ylabel='Number of T cells')
        self.barax2, self.barfig2 = plotGraphs.plotHorGrStBar(self.normtab, self.order, xticks=self.xticks, ylabel='Percent of T cells', ydim='%')
        #self.barax1, self.barfig1 = plotGraphs.plotHorGrStBar(self.table, self.order, xticks=self.xticks, ylabel='Number of T cells', hubysec=True)
        #self.barax2, self.barfig2 = plotGraphs.plotHorGrStBar(self.normtab, self.order, xticks=self.xticks, ylabel='Percent of T cells', ydim='%', hubysec=True)
        return
    
    def html(self, divname='orphan_tcr'):
        div = {
            'class': 'toggle_sub',
            'children': [
                {
                    'class': 'smallfig',
                    'uid': divname+'_1',
                    'figure': self.barfig1,
                    'innerHTML': ''
                },
                {
                    'class': 'smallfig',
                    'uid': divname+'_2',
                    'figure': self.barfig2,
                    'innerHTML': ''
                }
                ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'tcr_stats'
            }
        return argdic

class diverBar:
    def __init__(self, experiment):
        samples, groups, subgroup, celltypes, order = experiment.cellGroups['samples'], experiment.divGroup['SAMPLESPLIT'], 'clusters in '+experiment.maniFolds[0].label, experiment.maniFolds[0].annotations, experiment.sampleOrder 
        majorclone = []
        for k, v in experiment.cellGroups['major clonotypes'].iteritems():
            majorclone += v
        majorclone = set(majorclone)
        ttable, table, xticks = [], [], []
        if order == None:
            order = list(groups.keys())
        for sample in order:
            dat = groups[sample][subgroup]
            xticks += dat.keys()
            table.append([])
            ttable.append([])
        xticks = list(set(xticks))
        xticks.sort()
        for i in range(0, len(order)):
            sample = groups[order[i]][subgroup]
            for j in range(0, len(xticks)):
                if xticks[j] in sample:
                    val = sample[xticks[j]]
                else:
                    val = 0
                try:
                    val = float(val)
                except:
                    val = 0
                table[i].append(val)
            cells = set(samples[order[i]])
            for tick in xticks:
                val = 0
                for cell in majorclone:
                    if cell in cells:
                        if cell in celltypes:
                            if celltypes[cell] == tick:
                                val += 1
                ttable[i].append(val)
        self.order = order
        self.table = table
        self.ttable = ttable
        self.xticks = xticks
        self.ax1, self.fig1 = plotGraphs.plotGrDivBar(self.ttable, self.order, xticks=self.xticks, ylabel='Number of clonal cells')
        self.ax2, self.fig2 = plotGraphs.plotGrDivBar(self.table, self.order, xticks=self.xticks, ylabel='Shannon diversity score')
        return
    
    def html(self, divname='diversity_bar'):
        div = {
            'id': divname,
            'class': 'toggle_tab',

            'children': [
                {
                    'class': 'widefig',
                    'uid': divname+'_1',
                    'figure': self.fig2,
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                },
                {
                    'class': 'widefig',
                    'uid': divname+'_2',
                    'figure': self.fig1,
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                }
            ]
            }
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Diversity of samples',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class segmentBar:
    def __init__(self, experiment):
        samples, gdir, order = experiment.cellGroups['samples'], experiment.cellGroups, experiment.sampleOrder 
        labels = ['TRA V segment usage', 'TRB V segment usage', 'TRA D segment usage', 'TRB D segment usage', 'TRA J segment usage', 'TRB J segment usage']
        titles = [x.replace(' segment usage', '').replace(' ', '') for x in labels]
        self.figures = {}
        for i in range(0, len(labels)):
            table = []
            label, title = titles[i], labels[i]
            f = []
            for k, v in gdir[title].iteritems():
                f.append([k, len(v)])
            f.sort(key=lambda x: x[1], reverse=True)
            for sample in order:
                cells = set(samples[sample])
                row, xticks = [], []
                for a, b in f[:8]:
                    v = gdir[title][a]
                    val = len(set(v)&cells)
                    xticks.append(a+' ('+str(b)+')')
                    row.append(val)
                val = 0
                nval = 0
                for a, b in f[8:]:
                    v = gdir[title][a]
                    val += len(set(v)&cells)
                    nval += b
                if nval > 0:
                    xticks.append('Other ('+str(nval)+')')
                    row.append(val)
                table.append(row)
            ax, fig = plotGraphs.plotHorGrStBar(table, order, xticks=xticks, ylabel='Frequency of segment', title=title)
            self.figures[label] = fig
        self.order = order
        self.titles = titles
        return
    
    def html(self, divname='segment_bar'):
        div = {
            'id': divname,
            'class': 'toggle_tab',
            'children': []
            }
        for k in self.titles:
            v = self.figures[k]
            fd = {
                'class': 'smallfig',
                'uid': k,
                'figure': v,
                'innerHTML': ''
                }
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'VDJ segment usage',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic


class clonTopology:
    def __init__(self, experiment, ctypeorder=None):
        samples, clonogroup, tcrdat, manifolds, order = experiment.cellGroups['samples'], experiment.cellGroups['major clonotypes'], experiment.cellTCR, experiment.maniFolds, experiment.sampleOrder 
        table, figures = {}, {}
        selections = {'manifold': [], 'sample': ['all']}
        self.selorder = ['manifold', 'sample']
        self.fprefix = 'clone_map_'
        if order == None:
            order = list(samples.keys())
        clonok = list(clonogroup.keys())
        clonok.sort(key=lambda x: int(x.split('_')[1]))
        clonok = clonok[:12]
        for i, n, manifold in manifolds:
            selections['manifold'].append(n)
            coords = manifold.coordinates
            annot = manifold.annotations
            background = manifold.background
            celltypes = []
            bigcl, bigct, bigtc, bigmc = {}, {}, {}, {}
            for sample in order:
                sampletab = [{}, {}, {sample: [[], []]}]
                sabi = [[], []]
                if sample not in selections['sample']:
                    selections['sample'].append(sample)
                cells = set(samples[sample])
                for k in clonok:
                    if k not in bigcl: 
                        bigcl[k] = [[], []]
                    bcd = bigcl[k]
                    c_cells = set(clonogroup[k])&cells
                    xl, yl = [], []
                    for c in c_cells:
                        if c in coords:
                            x, y = coords[c]
                            xl.append(x)
                            yl.append(y)
                            sabi[0].append(x)
                            sabi[1].append(y)
                            bcd[0].append(x)
                            bcd[1].append(y)
                    sampletab[0][k] = [xl, yl]
                for k, v in annot.iteritems():
                    v = str(v)
                    if v not in sampletab[1]:
                        sampletab[1][v] = [[], []]
                        if v not in bigct:
                            bigct[v] = [[], []]
                            celltypes.append(v)
                    if k in cells:
                        cttab = sampletab[1][v]
                        bctab = bigct[v]
                        x, y = coords[k]
                        cttab[0].append(x)
                        cttab[1].append(y)
                        bctab[0].append(x)
                        bctab[1].append(y)
                for c in cells:
                    if c in tcrdat.cellTable:
                        ct = tcrdat.cellTable[c]['clonotype']
                        if tcrdat.clonotypeTable[ct]['cdrSym'] != '|||':
                            if c in coords:
                                x, y = coords[c]
                                sampletab[2][sample][0].append(x)
                                sampletab[2][sample][1].append(y)
                bigtc.update(sampletab[2])
                sabi = {sample: sabi}
                bigmc.update(sabi)
                table[self.fprefix+n+'_'+sample] = sampletab
            table[self.fprefix+n+'_all'] = [bigcl, bigct, bigtc]
            celltypes = list(set(celltypes))
            celltypes.sort()
            if ctypeorder == None:
                finalCTorer = celltypes
            else:
                finalCTorer = ctypeorder
            ax4, fig4 = plotGraphs.plotPojection(bigmc, catorder=order, title='Projection of clonal T cells on '+n, background=background)
            for sample in order:
                clontab, atab, ttab = table[self.fprefix+n+'_'+sample]
                ax1, fig1 = plotGraphs.plotPojection(clontab, catorder=clonok, title='Projection of clonotypes in '+sample+' on '+n, background=background)
                ax2, fig2 = plotGraphs.plotPojection(ttab, title='Projection of all T cells in '+sample+' on '+n, background=background)
                ax3, fig3 = plotGraphs.plotPojection(atab, catorder=finalCTorer, title='Projection of cell types in '+sample+' on '+n, background=background)
                figures[self.fprefix+n+'_'+sample] = [fig1, fig2, fig3, fig4]
            clontab, atab, ttab = table[self.fprefix+n+'_all']
            ax1, fig1 = plotGraphs.plotPojection(clontab, catorder=clonok, title='Projection of all cells in major clonotypes on '+n, background=background)
            ax2, fig2 = plotGraphs.plotPojection(ttab, catorder=order, title='Projection of all T cells in of all samples on '+n, background=background)
            ax3, fig3 = plotGraphs.plotPojection(atab, catorder=finalCTorer, title='Projection of cell types in all samples on '+n, background=background)
            figures[self.fprefix+n+'_all'] = [fig1, fig2, fig3, fig4]
        self.order = order
        self.selections = selections
        self.figures = figures
        return

    def html(self, divname='clone_map'):
        selectors = ''
        for k in self.selorder:
            v = self.selections[k]
            selectors += '<div class="selabel">'+k+'</div><select onchange="update_by_choice(this)">'
            for e in v:
                selectors += '<option value="'+e+'">'+e+'</option>'
            selectors += '</select>'
        div = {
            'id': divname,
            'class': 'toggle_tab',

            'children': [
                {
                    'class': 'chooser',
                    'id': divname+'_chooser',
                    'uid': divname+'_chooser',
                    'innerHTML': selectors + '<div class="breaker"></div>'
                }
            ]
            }
        for k, v in self.figures.iteritems():
            fd = {
                'id': k,
                'uid': k,
                'class': 'medfig',
                'children': []
                }
            for i in range(0, len(v)):
                cfd = {
                    'class': 'smallfig',
                    'uid': k+'_'+str(i),
                    'figure': v[i],
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                    }
                fd['children'].append (cfd)
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Cell projections',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class clonoDist:
    def __init__(self, experiment, celltypeorder=None):
        samples, celltypes, distances, order = experiment.cellGroups['samples'], experiment.maniFolds[0], experiment.cellDistances, experiment.sampleOrder
        self.table, self.selections, self.figures = [], [], {}
        cellabels, tlabels = {}, {}
        if celltypeorder == None:
            xticks = list(celltypes.cellTypes.keys())
            xticks.sort()
        else:
            xticks = celltypeorder
        for i in range(0, len(xticks)):
            tlabels[xticks[i]] = i
        for i in range(0, len(order)):
            sample = order[i]
            cells = set(samples[sample])
            for cell in cells:
                if cell in celltypes.annotations:
                    ct = tlabels[celltypes.annotations[cell]]
                else:
                    ct = len(xticks)
                cellabels[cell] = [i, ct, 'lightgray']
        collabels = [['Sample', order], ['Cell type', xticks]]
        for i, n, distance in distances:
            for cell, dat in cellabels.iteritems():
                dat[2] = 'lightgray'
            gd = distance.treeGroups
            go = list(gd.keys())
            go.sort(key=lambda x: int(x.split('Group ')[1]))
            for cell, dat in cellabels.iteritems():
                for j in range(0, len(go)):
                    if cell in gd[go[j]]:
                        dat[2] = j
            ncollabels = collabels + [['Cluster', go]]
            self.selections.append(n)
            self.figures['cell_distances_'+n] = [
                plotGraphs.plotCellTree(order, distance, cellabels, ncollabels, title='Clustering of clonotypes based on CDR3 sequence similarity ('+n+')'),
                plotGraphs.plotSimpleHeat(distance)
                ]
            self.table.append([i, n, distance, cellabels])
        return

    def html(self, divname='cell_distances'):
        selectors = '<div class="selabel">Property</div><select onchange="update_by_choice(this)">'
        for e in self.selections:
            selectors += '<option value="'+e+'">'+e+'</option>'
        selectors += '</select>'
        div = {
            'id': divname,
            'class': 'toggle_tab',
            'children': [
                {
                    'class': 'chooser',
                    'id': divname+'_chooser',
                    'uid': divname+'_chooser',
                    'innerHTML': selectors + '<div class="breaker"></div>'
                }
            ]
            }
        for k, v in self.figures.iteritems():
            fd = {
                'id': k,
                'uid': k,
                'class': 'medfig',
                'children': []
                }
            for i in range(0, len(v)):
                cfd = {
                    'class': 'widefig',
                    'uid': k+'_'+str(i),
                    'figure': v[i],
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                    }
                fd['children'].append (cfd)
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'TCR based cell clusters',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class cellTypeDist:
    def __init__(self, experiment):
        samples, celltypes, distances, order = experiment.cellGroups['samples'], experiment.maniFolds[0], experiment.cellDistances, experiment.sampleOrder
        self.table, self.selections, self.figures = [], [], {}
        cellabels, tlabels = {}, {}
        groupcolors = {}
        for i in range(0, len(order)):
            groupcolors[order[i]] = i
        for i, n, distance in distances:
            annot = celltypes.annotations
            groups = {}
            for sample in order:
                cells = set(samples[sample])
                for k, v in annot.iteritems():
                    if k in cells:
                        v = sample + ', ' + str(v)
                        if v not in groups:
                            groups[v] = []
                        groups[v].append(k)
            self.selections.append(n)
            self.figures['type_distances_'+n] = [plotGraphs.plotGrTree(distance, groups, groupcolors, title='Sequence similarity of TCRs in individual cell types')]
        return

    def html(self, divname='type_distances'):
        selectors = '<div class="selabel">Property</div><select onchange="update_by_choice(this)">'
        for e in self.selections:
            selectors += '<option value="'+e+'">'+e+'</option>'
        selectors += '</select>'
        div = {
            'id': divname,
            'class': 'toggle_tab',
            'children': [
                {
                    'class': 'chooser',
                    'id': divname+'_chooser',
                    'uid': divname+'_chooser',
                    'innerHTML': selectors + '<div class="breaker"></div>'
                }
            ]
            }
        for k, v in self.figures.iteritems():
            fd = {
                'id': k,
                'uid': k,
                'class': 'medfig',
                'children': []
                }
            for i in range(0, len(v)):
                cfd = {
                    'class': 'widefig',
                    'uid': k+'_'+str(i),
                    'figure': v[i],
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                    }
                fd['children'].append (cfd)
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Similarity of TCRs across cell types',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

class clonoGex:
    def __init__(self, experiment, genes=None, cellTypeOrder=None):
        samples, groups, manifolds, gex, order = experiment.cellGroups['samples'], experiment.cellGroups['clusters in seqSim'], experiment.maniFolds, experiment.cellGex.gex, experiment.sampleOrder
        self.order = order
        self.table = []
        self.figures = {}
        self.fprefix = 'gex_levels_'
        if order == None:
            order = list(samples.keys())
        if genes == None:
            self.genes = gex.genes
        collabels = []
        celltypes = []
        for gene in self.genes:
            table1, table2 = {'Sample': [], 'Group': [], 'No of reads': []}, {'Sample': [], 'Group': [], 'No of reads': []}
            for sample in order:
                scells = set(samples[sample])
                for group, gcells in groups.iteritems():
                    cells = scells&set(gcells)
                    for cell in cells:
                        reads = gex[cell][gene]
                        try:
                            reads = gex[cell][gene]
                            table1['Sample'].append(sample)
                            table1['Group'].append(group)
                            table1['No of reads'].append(reads)
                        except:
                            pass
                    collabels.append(group)
                m = manifolds[0]
                for c, d in m.annotations.iteritems():
                    if c in scells:
                        try:
                            reads = gex[c][gene]
                            table2['Sample'].append(sample)
                            table2['Group'].append(d)
                            table2['No of reads'].append(reads)
                        except:
                            pass
                    celltypes.append(d)
            self.table.append([gene, table1, table2])
        celltypes = list(set(celltypes))
        celltypes.sort()
        collabels = list(set(collabels))
        collabels.sort(key=lambda x: int(x.split('Group ')[1]))
        if cellTypeOrder == None:
            self.celltypeorder = celltypes
        else:
            self.celltypeorder = cellTypeOrder
        self.collabels = collabels
        for gene, table1, table2 in self.table:
            ax2, fig2 = plotGraphs.plotGexDistribution(table2, 'Group', 'No of reads', hue='Sample', title='Expression of ' + gene, order=self.celltypeorder, huerder=order, condensed=False, show_swarm=False, show_box=False, axisrotation=45)
            try:
                ax1, fig1 = plotGraphs.plotGexDistribution(table1, 'Group', 'No of reads', hue='Sample', title='Expression of ' + gene, order=self.collabels, huerder=order, condensed=False, show_swarm=False, show_box=False, axisrotation=45)
            except:
                ax1, fig1 = plotGraphs.emptyFig()
            try:
                ax2, fig2 = plotGraphs.plotGexDistribution(table2, 'Group', 'No of reads', hue='Sample', title='Expression of ' + gene, order=self.celltypeorder, huerder=order, condensed=False, show_swarm=False, show_box=False, axisrotation=45)
            except:
                ax2, fig2 = plotGraphs.emptyFig()
            #ax.set_ylim(bottom=0)
            self.figures[self.fprefix+gene] = [fig1, fig2]
        return

    def html(self, divname='gex_levels'):
        selectors = '<div class="selabel">gene</div><select onchange="update_by_choice(this)">'
        for e in self.genes:
            selectors += '<option value="'+e+'">'+e+'</option>'
        selectors += '</select>'
        div = {
            'id': divname,
            'class': 'toggle_tab',

            'children': [
                {
                    'class': 'chooser',
                    'id': divname+'_chooser',
                    'uid': divname+'_chooser',
                    'innerHTML': selectors + '<div class="breaker"></div>'
                }
            ]
            }
        for k, v in self.figures.iteritems():
            fd = {
                'id': k,
                'uid': k,
                'class': 'medfig',
                'children': []
                }
            for i in range(0, len(v)):
                cfd = {
                    'class': 'widefig',
                    'uid': k+'_'+str(i),
                    'figure': v[i],
                    'figargs': [None, None, 300],
                    'innerHTML': ''
                    }
                fd['children'].append (cfd)
            div['children'].append (fd)
        argdic = {
            'name': divname,
            'div': div,
            'parent': 'content',
            'tempstore': True,
            'sidelabel': [
                'side_bar', 'button_'+divname,
                {
                    'innerHTML': 'Expression of top genes',
                    'class': 'menu_item',
                    'onclick': "toggle_visibility(this, '"+divname+"')"
                }
                ]
            }
        return argdic

if __name__ == '__main__':
    __main__()