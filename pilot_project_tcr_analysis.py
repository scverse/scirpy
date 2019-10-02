#!/usr/bin/python
# -*- coding: utf-8 -*-

from scTCRpy.utils import stampTime
from scTCRpy.layoutExp import eXperiment
from scTCRpy.reportHTML import htmlReporter
from scTCRpy import talkR
from scTCRpy import resultData

def __main__():
    sample_features = [
        ['', 'Sample1', 'Sample_1_', 'pilot_VDJ1/outs', 'pilot_count1/call-count/shard-0/execution/pilot_count1_lane1/outs'],
        ['', 'Sample2', 'Sample_2_', 'pilot_VDJ2/outs', 'pilot_count2/call-count/shard-0/execution/pilot_count2_lane1/outs'],
        ['', 'Sample3', 'Sample_3_', 'pilot_VDJ3/outs', 'pilot_count3/call-count/shard-0/execution/pilot_count3_lane1/outs'],
        ['', 'Sample4', 'Sample_4_', 'pilot_VDJ4/outs', 'pilot_count4/call-count/shard-0/execution/pilot_count4_lane1/outs']
        ]
    starttime = stampTime('Starting execution at')
    #Create an empty layout structure, set top folder for files and initialize main data containers (also offering a possibility to pass prepopulated data containers form upstream steps)
    experiment = eXperiment(projectRoot='/home/singlecell/singleCellSeq/results/Giorgos/pilot_project', outDir='tcr_out', reportFile='TCR_pilot_report.html', genListFile='integrate/genelist.txt', criticalClonotypeSize=6, verbose=True)
    #Add samples to the experiment by defining name, barcode prefix and file paths needed for downstream steps
    preprocesstime = stampTime('Preprocessing sample data')
    for a1, a2, a3, a4, a5 in sample_features:
        experiment.addSample(a1, a2, a3, t_tcrDir=a4, t_gexDir=a5)
        timepoint = stampTime('Parsed basic sample info.', before=preprocesstime)
    #Reassign clonotypes based on the dominant TCR chain pair
    experiment.cellTCR.reassessClonotypes()
    timepoint = stampTime('Reassigned clonotypes.', before=timepoint)
    #Add some two-dimensional projections of cell populations - primarily based on gene expression, but theoratically any coordinate collection can be imported
    experiment.addProjection('cells', proFile='integrate/celltypes.txt')
    experiment.addProjection('tSNE', coordFile='integrate/tsne.csv', annotFile='integrate/a_tsne.tsv')
    experiment.addProjection('umap', coordFile='integrate/umap.csv', annotFile='integrate/a_umap.tsv')
    timepoint = stampTime('Cell population projections added.', before=timepoint)
    #A possible method to claculate distances is the sequence similarity of the CDR3 regions, a method reimplemented from tcrDist
    experiment.distanceCalc('seqSim', metric='TCR')
    timepoint = stampTime('Distance matrix based on TCR sequences calculated.', before=timepoint)
    timepoint = stampTime('First round of data preprocessing finised.', before=preprocesstime)
    #Export data to tables for R scripts and other third party analysis. while recollecting diversity and Kidera calculations from R
    talkrtime = stampTime('Starting data exchange with R.')
    talkR.featRchain(experiment)
    talkR.exportCellFeatures(experiment)
    talkR.groupDiversity(experiment)
    stampTime('Diversity scores and Kidera factors called with R scripts.', before=talkrtime)
    #Another distance matrix could also be calculated, based on Kidera factors of the CDR3 regions of the dominant TCR pair
    experiment.distanceCalc('Kidera', metric='Kidera') #Infinite distances after calculation, fix it later!
    timepoint = stampTime('Distance matrix based on Kidera factors calculated.', before=timepoint)
    #After all data is preprocessed, final calculations and visualizations can be done
    reporttime = stampTime('Starting to plot graphs for the report file.')
    reporter = htmlReporter(experiment.reportFile, template='/home/singlecell/scripts/Tamas/scTCRpy/report_template.html', saveres=150, savepics=True)
    #Some intital setup of html is needed
    html_frame = [
        ('title', {'div': {'class': 'main_title', 'innerHTML': 'Analysis of TCR sequencing results'}}),
        ('side_bar', {'div': {'class': 'side_content'}}),
        ('content', {'div': {'class': 'main_content'}})
        ]
    reporter.add(html_frame)
    timepoint = stampTime('Report file initialized', before=timepoint)
    #Add a clonotype frequency table
    clonotype_table = resultData.clonoTable(experiment)
    reporter.add(**clonotype_table.html())
    timepoint = stampTime('Clonotype tables created for the report', before=reporttime)
    #Plot the overlap between clonotypes of the two samples
    clonoverlap = (resultData.clonOverlap(experiment))
    reporter.add(**clonoverlap.html())
    timepoint = stampTime('Clonotype overlap plotted', before=timepoint)
    #Add a tab for basic TCR statistics
    tcr_stats = {
        'div': {
            'class': 'toggle_tab',
            'id': 'tcr_stats'
            },
        'parent': 'content',
        'sidelabel': [
            'side_bar', 'button_tcr_stats',
            {
                'innerHTML': 'TCR chain statistics',
                'class': 'menu_item',
                'onclick': "toggle_visibility(this, 'tcr_stats')"
            }
            ]
        }
    reporter.add('tcr_stats', **tcr_stats)
    #Plot CDR3 amino acid lengths
    lena_tcr = resultData.lenCdrAa(experiment)
    reporter.add(**lena_tcr.html())
    timepoint = stampTime('CDR3 amino acid lengths plotted', before=timepoint)
    #Plot CDR3 nucleotide lengths
    lent_tcr = resultData.lenCdrNt(experiment)
    reporter.add(**lent_tcr.html())
    timepoint = stampTime('CDR3 nucleotide lengths plotted', before=timepoint)
    #Plot added nucleotides
    adn_tcr = resultData.lenCdrIns(experiment)
    reporter.add(**adn_tcr.html())
    timepoint = stampTime('CDR3 added nucleotides plotted', before=timepoint)
    #Plot TCR chain expression
    xpr_tcr = resultData.exprTVio(experiment)
    reporter.add(**xpr_tcr.html())
    timepoint = stampTime('TCR chain expression levels plotted', before=timepoint)
    #Plot TCR chain pairing in samples and cell types
    orhpan_tcr = resultData.orphanBar(experiment)
    reporter.add(**orhpan_tcr.html())
    #Show VDJ segment usage and frequency of segments
    seg_use = resultData.segmentBar(experiment)
    reporter.add(**seg_use.html())
    timepoint = stampTime('VDJ segment usage plotted', before=timepoint)
    #Plot clonotypes and cell types over a projection of cell features (mainly cell types, based on gene expression)
    cl_project = resultData.clonTopology(experiment)
    reporter.add(**cl_project.html())
    timepoint = stampTime('Clonotype projection plotted', before=timepoint)
    #Show diversity of samples and cell types
    group_div = resultData.diverBar(experiment)
    reporter.add(**group_div.html())
    timepoint = stampTime('Diversity of sample groups plotted', before=timepoint)
    timepoint = stampTime('TCR chain pairing stats plotted', before=timepoint)
    #Show how the cells clustered based on distance metrics
    cell_dist = resultData.clonoDist(experiment)
    reporter.add(**cell_dist.html())
    timepoint = stampTime('T cells clustered based on TCR similarity', before=timepoint)
    #Plot TCR based distances between cell types
    type_dist = resultData.cellTypeDist(experiment)
    reporter.add(**type_dist.html())
    timepoint = stampTime('T cell types clustered based on TCR similarity', before=timepoint)
    #Plot the expression of top genes in clonotype groups
    top_gex = resultData.clonoGex(experiment)
    reporter.add(**top_gex.html())
    timepoint = stampTime('Expression of top genes plotted', before=timepoint)
    reporter.report()
    stampTime('Report compiled to HTML file ' + experiment.reportFile, before=reporttime)
    stampTime('TCR results successfully analysed, script execution ends.', before=starttime)
    return

if __name__ == '__main__':
    __main__()
