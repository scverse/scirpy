import pandas as pd
import numpy as np
import scirpy as ir 
import anndata
import scanpy as sc

class ImportBDrhapsodyData:

    def load_tcr_seq(data):

        tcr_table = pd.read_csv(data, sep=",", index_col=None, na_values=["None"], true_values=["True"],comment='#')
        tcr_table['productive_Alpha_Gamma'] = tcr_table.TCR_Alpha_Gamma_CDR3_Translation_Dominant.str.contains('\*', regex=True)
        tcr_table['productive_Beta_Delta'] = tcr_table.TCR_Beta_Delta_CDR3_Translation_Dominant.str.contains('\*', regex=True)
        print(tcr_table['productive_Beta_Delta'])
        tcr_table
        tcr_cells = []
        for idx, row in tcr_table.iterrows():
            cell = ir.io.AirrCell(cell_id=row["Cell_Index"])
            alpha_chain = ir.io.AirrCell.empty_chain_dict()
            beta_chain = ir.io.AirrCell.empty_chain_dict()
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

        data_tcr = ir.io.from_airr_cells(tcr_cells)

        return data_tcr

    def load_dCODE_data(data):

        data_decode = pd.read_csv(data, index_col=0, comment='#')

        return data_decode
    

    def load_rna_seq_data(data):

        data_rna = pd.read_csv(data, index_col=0, comment='#')

        return data_rna


    def create_adata_from_rhapsody(rna,decode=1,tcr=1):

        data_rna = ImportBDrhapsodyData.load_rna_seq_data(data=rna)
        genes = list(data_rna.columns)

        if decode != 1:
            data_decode = ImportBDrhapsodyData.load_dCODE_data(data=decode)
        
        else:
            pass

        data_decode = data_decode.drop(columns=genes)

        try:
            obs = data_decode
        except:
            pass
        try:
            adata = sc.AnnData(np.array(data_rna),obs=obs,var=genes)
        except:
            adata = sc.AnnData(np.array(data_rna),var=genes)
        
        if tcr != 1:
            data_tcr = ImportBDrhapsodyData.load_tcr_seq(data=tcr)
            ir.pp.merge_with_ir(adata, data_tcr)
        else:
            pass

        return adata
