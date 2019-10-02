#!/usr/bin/env nextflow

nextflow.preview.dsl=2

process run_tcr {
    conda "envs/tcr.yml"

    input:
        file "data"
        file "samples.tsv"
        file "celltypes.tsv"
        file "genelist.txt"
        file "template.html"

    output:
        file "TCR_report.html"

    publishDir params.project_home

    """
    basic_tcr.py \
        .  \
        out \
        samples.tsv \
        celltypes.tsv \
        genelist.txt \
        TCR_report.html \
        template.html
    """
}

workflow {
    run_tcr(
        file(params.data_dir),
        file(params.tcr_samples),
        file(params.cell_type_annotation),
        file(params.genelist),
        file(params.config.report_template)
    )
}
