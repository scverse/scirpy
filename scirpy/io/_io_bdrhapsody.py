import pandas as pd
from ._datastructures import AirrCell
from ._convert_anndata import from_airr_cells
from ..pp._merge_adata import merge_with_ir

def read_bd_vdj_csv(data,adata):

    tcr_table = pd.read_csv(data, sep=",", index_col=None, na_values=["None"], true_values=["True"],comment='#')
    tcr_table['productive_Alpha_Gamma'] = tcr_table.TCR_Alpha_Gamma_CDR3_Translation_Dominant.str.contains('\*', regex=True)
    tcr_table['productive_Beta_Delta'] = tcr_table.TCR_Beta_Delta_CDR3_Translation_Dominant.str.contains('\*', regex=True)
    print(tcr_table['productive_Beta_Delta'])
    tcr_table
    tcr_cells = []
    for idx, row in tcr_table.iterrows():
        cell = AirrCell(cell_id=row["Cell_Index"])
        alpha_chain = AirrCell.empty_chain_dict()
        beta_chain = AirrCell.empty_chain_dict()
        alpha_chain.update(
            {
                "locus": "TRA",
                "junction_aa": row["TCR_Alpha_Gamma_CDR3_Translation_Dominant"],
                "junction": row["TCR_Alpha_Gamma_CDR3_Nucleotide_Dominant"],
                "consensus_count": row["TCR_Alpha_Gamma_Read_Count"],
                "duplicate_count" : row["TCR_Alpha_Gamma_Molecule_Count"],
                "v_call": row["TCR_Alpha_Gamma_V_gene_Dominant"],
                "j_call": row["TCR_Alpha_Gamma_J_gene_Dominant"],
                "productive": row["productive_Alpha_Gamma"]
            }
        )
        beta_chain.update(
            {
                "locus": "TRB",
                "junction_aa": row["TCR_Beta_Delta_CDR3_Translation_Dominant"],
                "junction": row["TCR_Beta_Delta_CDR3_Nucleotide_Dominant"],
                "consensus_count": row["TCR_Beta_Delta_Read_Count"],
                "duplicate_count" : row["TCR_Beta_Delta_Molecule_Count"],
                "v_call": row["TCR_Beta_Delta_V_gene_Dominant"],
                "d_call": row["TCR_Beta_Delta_D_gene_Dominant"],
                "j_call": row["TCR_Beta_Delta_J_gene_Dominant"],
                "productive": row["productive_Beta_Delta"]
            }
        )
        cell.add_chain(alpha_chain)
        cell.add_chain(beta_chain)
        tcr_cells.append(cell)

    data_tcr = from_airr_cells(tcr_cells)
    merge_with_ir(adata, data_tcr)

    return adata
