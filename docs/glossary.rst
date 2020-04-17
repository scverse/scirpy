Glossary
========

.. glossary::

    TCR
        T-cell receptor. TODO should contain description of purpose and illustration. 

    Clonotype
        A clonotype designates a collection of T or B cells that bear the same adaptive
        immune receptors, and thus recognize the same epitopes. Generally, these cells are 
        also descendants of a common, antecedent cell and belong to the same cell clone.
        In single-cell RNA-sequencing (scRNA-seq) data, T cells sharing identical
        complementarity-determining regions 3 (CDR3) sequences of both α and β TCR chains 
        make up a clonotype.

    Private clonotype

    Public clonotype

    CDR3

    VDJ
        
    Dual TCRs 
        Contrary to what would be expected based on the previously described mechanism of
        allelic exclusion (:cite:`Brady2010-gh`), scRNA-seq datasets can feature a considerable
        number of cells with more than one TCR α and β pair. Since cells with more than one 
        productive CDR3 sequence for each chain did not fit into our understanding of 
        T cell biology, most TCR analysis tools ignore these cells (:cite:`Fischer2019`, 
        :cite:`Zhang2018-ip`), or select the CDR3 sequence with the highest expression level
        (:cite:`Afik2017-sg`). While in some cases these double-TCR cells might represent 
        artifacts (e.g. doublets of a CD8+ and a CD4+ T cell engaged in an immunological 
        synapse), there is an increasing amount of evidence in support of a bone fide
        dual-TCR population (:cite:`Schuldt2019`, :cite:`Ji2010-bn`).

        Scirpy allows investigating the composition and phenotypes of both single- and dual-TCR 
        T cells by leveraging a T cell model similar to the one proposed in 
        :cite:`Stubbington2016-kh`, where T cells are allowed to have a primary and a secondary 
        pair of α- and β chains. For each cell, the primary pair consists of the α- and β-chain 
        with the highest read count. Likewise, the secondary pair is the pair of α/β-chains with
        the second highest expression level. Based on the assumption that each cell has only two
        copies of the underlying chromosome set, if more than two variants of a chain are 
        recovered for the same cell, the excess TCR chains are ignored by Scirpy and the 
        corresponding cells flagged as “multichain”. This filtering strategy leaves the choice 
        of discarding or including multichain cells in downstream analyses.

    Multichain

