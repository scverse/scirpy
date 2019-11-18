#!/usr/bin/env nextflow
params.packageDir = "$workflow.projectDir/lib"
params.condaEnvPy = 'envs/tcrpy3.yml'
params.condaEnvR = 'envs/tcrpy_r.yml'


Channel
    .fromPath(params.contigFiles)
    .merge(Channel.fromPath(params.consensusFiles))
    .merge(Channel.from(params.barcodePrefixes))
    .set {inputData}



process parseVDJresults {
    conda params.condaEnvPy

    input:
        set file(contigFiles), file(consensusFiles), val(barcodePrefixes) from inputData

    output:
        file 'cells.tsv' into sampleCellTable
        file 'chains.tsv' into sampleChainTable
        file 'cdrs.tsv' into sampleCDR3Table

    script:
        """
        parseVDJ.py $barcodePrefixes $contigFiles $consensusFiles cells.tsv chains.tsv cdrs.tsv
        """
}


sampleCellTable.collectFile(name: 'mergedCells.tsv', keepHeader: true).set{mergedCellTable}
sampleCDR3Table.collectFile(name: 'mergedCDRs.tsv').set{mergedCDR3Table}
sampleChainTable.collectFile(name: 'mergedChains.tsv').set{mergedChainTable}


process callClonotypes {

    conda params.condaEnvPy

    publishDir params.outDir, mode: 'copy', pattern: '{clonotypeTable.tsv, chainNet.tsv}'

    input:
        file mergedCDR3Table

    output:
        file 'clonotypeTable.tsv' into mappedClonotypes
        file 'additionalCellInfo.tsv' into additionalCellInfo
        file 'chainConvergence.tsv' into chainConvergence
        file 'chainMap.tsv' into chainMap
        file 'chainPairs.tsv' into chainPairs1, chainPairs2, chainPairs3
        file 'chainNet.tsv' into chainNet
        file 'inToDiv.txt' into inToDiv
        file 'inToDist.txt' into inToDis, inToKid

    script:
        """
        callClonotype.py $mergedCDR3Table clonotypeTable.tsv additionalCellInfo.tsv chainConvergence.tsv chainMap.tsv chainPairs.tsv chainNet.tsv inToDiv.txt inToDist.txt
        """
}

process calcKidera {
    conda params.condaEnvR

    input:
        file inToKid

    output:
        file 'chainKideras.tsv' into kideraTable1, kideraTable2

    script:
        """
        kideraCalc.r $inToKid chainKideras.tsv ${params.packageDir}
        """
}

process calcChainDiversity {
    conda params.condaEnvR

    input:
        file inToDiv

    output:
        file 'chainDiv.tsv' into divTable

    script:
        """
        chainDiversityCalc.r $inToDiv chainDiv.tsv ${params.packageDir}
        """
}

process cdrDistances {

    cpus params.nCpus
    conda params.condaEnvPy
    publishDir params.outDir, mode: 'copy', pattern: 'cdrSeqDistanceMatrix.h5'

    input:
        file inToDis
        file chainPairs1

    output:
        file 'cdrSeqDistanceMatrix.h5' into cdrDistances

    when:
        params.includeCDRdistances

    script:
        """
        chainBasedCellDistanceCalculations.py cdrseq $inToDis \
            cdrSeqDistanceMatrix.h5 ${task.cpus}
        chainBasedCellDistanceCalculations.py celldist \
            cdrSeqDistanceMatrix.h5 $chainPairs1 \
            ${params.distanceDisambiguation} ${task.cpus}
        """
}

process kideraDistances {

    cpus params.nCpus
    conda params.condaEnvPy
    publishDir params.outDir, mode: 'copy', pattern: 'kideraDistanceMatrix.h5'

    input:
        file kideraTable1
        file chainPairs2

    output:
        file 'kideraDistanceMatrix.h5' into kideraDistances

    script:
        """
        chainBasedCellDistanceCalculations.py kidera \
            $kideraTable1 kideraDistanceMatrix.h5
        chainBasedCellDistanceCalculations.py celldist \
            kideraDistanceMatrix.h5 $chainPairs2 \
            ${params.distanceDisambiguation} ${task.cpus}
        """
}

process summarizeChainData {

    conda params.condaEnvPy
    publishDir params.outDir

    input:
        file chainPairs3
        file divTable
        file kideraTable2
        file chainMap
        file mergedChainTable

    output:
        file 'chainTable.tsv' into sumChainTab

    script:
        """
        summarizeReceptorChainTables.py $chainPairs3 $divTable \
            $kideraTable2 $chainMap $mergedChainTable \
            receptorTable.tsv chainTable.tsv
        """
}

process summarizeCellData {
    conda params.condaEnvPy

    publishDir params.outDir

    input:
        file mergedCellTable
        file additionalCellInfo

    output:
        file 'cellTable.tsv' into sumCellTab

    script:
        """
        summarizeCellTable.py $mergedCellTable $additionalCellInfo cellTable.tsv
        """
}
