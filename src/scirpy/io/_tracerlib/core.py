"""Minimal port of `tracerlib`
to enable loading of pickled objects
from the Tracer output.

Originally published at https://github.com/Teichlab/tracer
under Apache 2.0 License.

Copyright (c) 2015 EMBL - European Bioinformatics Institute
Modified 2020 - Gregor Sturm

Licensed under the Apache License, Version 2.0 (the
"License"); you may not use this file except in
compliance with the License.  You may obtain a copy of
the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
"""

from collections import Counter, defaultdict

import six

# from Bio.Alphabet import generic_dna
# from Bio.Seq import Seq
# import pdb


class Cell:
    """Class to describe T cells containing A and B loci"""

    def __init__(
        self,
        cell_name,
        recombinants,
        is_empty=False,
        species="Mmus",
        receptor=None,
        loci=None,
    ):
        self.name = cell_name
        self.bgcolor = None
        self.recombinants = self._process_recombinants(recombinants, receptor, loci)
        self.is_empty = self._check_is_empty()
        self.species = species
        # self.cdr3_comparisons = {'A': None, 'B': None, 'mean_both': None}
        # invariant_types = []
        # if invariant_cells is not None:
        #    for ic in invariant_cells:
        #        itype = ic.check_for_match(self)
        #        if itype is not None:
        #            invariant_types.append(itype)

        # self.is_inkt = self._check_if_inkt()

    def _process_recombinants(self, recombinants, receptor, loci):
        recombinant_dict = defaultdict(dict)
        if recombinants is not None:
            for r_name, r in six.iteritems(recombinants):
                r_name = r_name.split("_")
                receptor = r_name[0]
                locus = r_name[1]
                recombinant_dict[receptor][locus] = r

        # normalise this to put None in cases where no receptors found
        for l in loci:
            if l not in recombinant_dict[receptor]:
                recombinant_dict[receptor][l] = None
        return dict(recombinant_dict)

    def _check_is_empty(self):
        if self.recombinants is None or len(self.recombinants) == 0:
            return True
        else:
            return False

    def missing_loci_of_interest(self, receptor_name, loci):
        recombinants = self.recombinants[receptor_name]
        loci_of_interest = set(loci)
        loci_in_cell = set()
        for l in loci:
            if l in recombinants and (recombinants[l] is not None and len(recombinants[l]) > 0):
                loci_in_cell.add(l)
        if len(loci_of_interest.intersection(loci_in_cell)) == 0:
            return True
        else:
            return False

    # def _check_if_inkt(self):
    #    A_recombs = self.getMainRecombinantIdentifiersForLocus("A")
    #    inkt_ident = False
    #    for recomb in A_recombs:
    #        for invar_seq in self.invariant_seqs:
    #            if invar_seq['V'] in recomb and invar_seq['J'] in recomb:
    #                inkt_ident = recomb
    #    return (inkt_ident)

    # def reset_cdr3_comparisons(self):
    #    self.cdr3_comparisons = {'A': None, 'B': None, 'mean_both': None}

    def getAllRecombinantIdentifiersForLocus(self, locus):
        recombinants = self.all_recombinants[locus]
        identifier_list = set()
        if recombinants is not None:
            for recombinant in recombinants:
                all_possible_recombinant_identifiers = recombinant.all_poss_identifiers
                for identifier in all_possible_recombinant_identifiers:
                    identifier_list.add(identifier)
        return identifier_list

    def getMainRecombinantIdentifiersForLocus(self, receptor_name, locus):
        recombinants = self.recombinants[receptor_name][locus]
        identifier_list = set()
        if recombinants is not None:
            for recombinant in recombinants:
                identifier_list.add(recombinant.identifier)
        return identifier_list

    # def getAllRecombinantCDR3ForLocus(self, locus):
    #    recombinants = self.all_recombinants[locus]
    #    identifier_list = set()
    #    if recombinants is not None:
    #        for recombinant in recombinants:
    #            cdr3 = str(recombinant.cdr3)
    #            if "Couldn't" not in cdr3:
    #                identifier_list.add(cdr3)
    #    return (identifier_list)

    def html_style_label_dna(self, receptor, loci, colours):
        # colours = {'A': {'productive': '#E41A1C', 'non-productive': "#ff8c8e"},
        #           'B': {'productive': '#377eb8', 'non-productive': "#95c1e5"},
        #           'G': {'productive': '#4daf4a', 'non-productive': "#aee5ac"},
        #           'D': {'productive': '#984ea3', 'non-productive': "#deace5"}}
        # locus_names = ['A', 'B', 'G', 'D']

        recombinants = {}
        final_string = '<<FONT POINT-SIZE="16"><B>' + self.name + "</B></FONT>"
        for locus, recombinant_list in six.iteritems(self.recombinants[receptor]):
            recombinant_set = set()
            if recombinant_list is not None:
                for recombinant in recombinant_list:
                    if recombinant.productive:
                        i = 0
                    else:
                        i = 1
                    recombinant_set.add(
                        "<BR/>" + f'<FONT COLOR = "{colours[receptor][locus][i]}">' + recombinant.identifier + "</FONT>"
                    )

                recombinants[locus] = recombinant_set
        for locus in loci:
            if locus in recombinants.keys():
                id_string = "".join(recombinants[locus])
                final_string = final_string + id_string
        final_string = final_string + ">"
        return final_string
        # return(self.name)

    def html_style_label_for_circles(self, receptor, loci, colours):
        # colours = {'A': {'productive': '#E41A1C', 'non-productive': "#ff8c8e"},
        #           'B': {'productive': '#377eb8', 'non-productive': "#95c1e5"},
        #           'G': {'productive': '#4daf4a', 'non-productive': "#aee5ac"},
        #           'D': {'productive': '#984ea3', 'non-productive': "#deace5"}}
        # locus_names = ['A', 'B', 'G', 'D']

        recombinants = {}
        final_string = '<<table cellspacing="6px" border="0" cellborder="0">'
        # final_string = "<"
        for locus, recombinant_list in six.iteritems(self.recombinants[receptor]):
            recombinant_set = []
            if recombinant_list is not None:
                for recombinant in recombinant_list:
                    if recombinant.productive:
                        i = 0
                    else:
                        i = 1
                    recombinant_set.append(
                        f'<tr><td height="10" width="40" bgcolor="{colours[receptor][locus][i]}"></td></tr>'
                    )

                recombinants[locus] = recombinant_set
        strings = []
        for locus in loci:
            if locus in recombinants.keys():
                strings.append("".join(recombinants[locus]))

        id_string = "".join(strings)
        final_string = final_string + id_string
        final_string = final_string + "</table>>"
        return final_string

    def __str__(self):
        return self.name

    def full_description(self):
        # pdb.set_trace()
        return_list = [self.name, "#TCRA#"]

        if self.A_recombinants is not None:
            for recombinant in self.A_recombinants:
                return_list.append(str(recombinant))
        else:
            return_list.append("No TCRA recombinants")

        return_list.append("\n#TCRB#")
        if self.B_recombinants is not None:
            for recombinant in self.B_recombinants:
                return_list.append(str(recombinant))
        else:
            return_list.append("No TCRB recombinants")

        return_list.append("\n#TCRG#")
        if self.G_recombinants is not None:
            for recombinant in self.G_recombinants:
                return_list.append(str(recombinant))
        else:
            return_list.append("No TCRG recombinants")

        return_list.append("\n#TCRD#")
        if self.D_recombinants is not None:
            for recombinant in self.D_recombinants:
                return_list.append(str(recombinant))
        else:
            return_list.append("No TCRD recombinants")

        return "\n".join(return_list)

    def get_fasta_string(self):
        seq_string = []

        for receptor, locus_dict in six.iteritems(self.recombinants):
            for locus, recombinants in six.iteritems(locus_dict):
                if recombinants is not None:
                    for rec in recombinants:
                        name = f">TRACER|{receptor}|{locus}|{rec.contig_name}|{rec.identifier}"
                        seq = rec.dna_seq
                        seq_string.append("\n".join([name, seq]))

        # for locus, recombinants in six.iteritems(self.all_recombinants):
        #    if recombinants is not None:
        #        for rec in recombinants:
        #            name = ">TCR|{contig_name}|{identifier}".format(contig_name=rec.contig_name,
        #                                                            identifier=rec.identifier)
        #            seq = rec.dna_seq
        #            seq_string.append("\n".join([name, seq]))
        return "\n".join(seq_string + ["\n"])

    def summarise_productivity(self, receptor, locus):
        if (
            self.recombinants is None
            or locus not in self.recombinants[receptor]
            or self.recombinants[receptor][locus] is None
        ):
            return "0/0"
        else:
            recs = self.recombinants[receptor][locus]
            prod_count = 0
            total_count = len(recs)
            for rec in recs:
                if rec.productive:
                    prod_count += 1
            return f"{prod_count}/{total_count}"

    def filter_recombinants(self):
        for receptor, locus_dict in six.iteritems(self.recombinants):
            for locus, recombinants in six.iteritems(locus_dict):
                if recombinants is not None:
                    if len(recombinants) > 2:
                        TPM_ranks = Counter()
                        for rec in recombinants:
                            TPM_ranks.update({rec.contig_name: rec.TPM})
                        two_most_common = [x[0] for x in TPM_ranks.most_common(2)]
                        to_remove = []
                        for rec in recombinants:
                            if rec.contig_name not in two_most_common:
                                to_remove.append(rec)
                        for rec in to_remove:
                            self.recombinants[receptor][locus].remove(rec)

    def count_productive_recombinants(self, receptor, locus):
        recs = self.recombinants[receptor][locus]
        count = 0
        if recs is not None:
            for rec in recs:
                if rec.productive:
                    count += 1
        return count

    def count_total_recombinants(self, receptor, locus):
        recs = self.recombinants[receptor][locus]
        count = 0
        if recs is not None:
            count = len(recs)
        return count

    def get_trinity_lengths(self, receptor, locus):
        recs = self.recombinants[receptor][locus]
        lengths = []
        if recs is not None:
            for rec in recs:
                lengths.append(len(rec.trinity_seq))
        return lengths

    def has_excess_recombinants(self, max_r=2):
        for _receptor, locus_dict in six.iteritems(self.recombinants):
            for _locus, recs in six.iteritems(locus_dict):
                if recs is not None:
                    if len(recs) > max_r:
                        return True


class Recombinant:
    """Class to describe a recombined TCR locus as determined from the single-cell pipeline"""

    def __init__(
        self,
        contig_name,
        locus,
        identifier,
        all_poss_identifiers,
        productive,
        stop_codon,
        in_frame,
        TPM,
        dna_seq,
        hit_table,
        summary,
        junction_details,
        best_VJ_names,
        alignment_summary,
        trinity_seq,
        imgt_reconstructed_seq,
        has_D,
        cdr3nt,
        cdr3,
    ):
        self.contig_name = contig_name
        self.locus = locus
        self.identifier = identifier
        self.all_poss_identifiers = all_poss_identifiers
        self.productive = productive
        self.TPM = TPM
        self.dna_seq = dna_seq
        self.hit_table = hit_table
        self.summary = summary
        self.junction_details = junction_details
        self.best_VJ_names = best_VJ_names
        self.alignment_summary = alignment_summary
        self.in_frame = in_frame
        self.stop_codon = stop_codon
        self.trinity_seq = trinity_seq
        self.imgt_reconstructed_seq = imgt_reconstructed_seq
        self.has_D_segment = has_D
        self.cdr3nt = cdr3nt
        self.cdr3 = cdr3

    def __str__(self):
        return f"{self.identifier} {self.productive} {self.TPM}"

    def get_summary(self):
        summary_string = f"##{self.contig_name}##\n"
        if not self.has_D_segment:
            V_segment = self.summary[0]
            J_segment = self.summary[1]
            segments_string = f"V segment:\t{V_segment}\nJ segment:\t{J_segment}\n"
        else:
            V_segment = self.summary[0]
            D_segment = self.summary[1]
            J_segment = self.summary[2]
            segments_string = f"V segment:\t{V_segment}\nD segment:\t{D_segment}\nJ segment:\t{J_segment}\n"
        summary_string += segments_string
        summary_string += f"ID:\t{self.identifier}\n"
        summary_string += (
            f"TPM:\t{self.TPM}\nProductive:\t{self.productive}\nStop codon:"
            f"\t{self.stop_codon}\nIn frame:\t{self.in_frame}\n"
        )

        # lowercase CDR3 sequences if non-productive
        cdr3 = self.cdr3
        cdr3nt = self.cdr3nt
        if not self.productive:
            cdr3 = cdr3.lower()
            cdr3nt = cdr3nt.lower()

        summary_string += f"CDR3aa:\t{cdr3}\nCDR3nt:\t{cdr3nt}\n\n"

        summary_string += (
            "Segment\tquery_id\tsubject_id\t% identity\t"
            "alignment length\tmismatches\tgap opens\tgaps"
            "\tq start\tq end\ts start\ts end\te value\tbit score\n"
        )
        for line in self.hit_table:
            summary_string = summary_string + "\t".join(line) + "\n"
        return summary_string


class Invar_cell:
    """Class to describe invariant cells and their specific sequences"""

    def __init__(self, d):
        self.name = d["cell_name"]
        self.receptor_type = d["receptor_type"]
        self.invariant_recombinants = d["recombinants"]
        self.defining_locus = d["defining_locus"]
        self.expected_string = self._get_expected_string()

    def check_for_match(self, cell, locus):
        found_identifiers = set()
        found_locus = False

        # check for expected recombinants for defining locus
        cell_recs = cell.recombinants[self.receptor_type][locus]
        invariant_recs = self.invariant_recombinants[locus]
        if cell_recs is not None:
            for rec in cell_recs:
                if rec.productive:
                    for ident in rec.all_poss_identifiers:
                        ident = ident.split("_")
                        v = ident[0]
                        j = ident[2]
                        for ivr in invariant_recs:
                            if (v == ivr["V"] or ivr["V"] == "*") and (j == ivr["J"] or ivr["J"] == "*"):
                                found_locus = True
                                found_identifiers.add("_".join(ident))

        return found_locus, found_identifiers

    def _get_expected_string(self):
        s = ""
        defining_recs = self.invariant_recombinants[self.defining_locus]
        r = defining_recs[0]
        s = s + "-".join([r["V"], r["J"]])
        if len(defining_recs) > 1:
            for r in defining_recs[1:]:
                s = s + " | " + "-".join([r["V"], r["J"]])

        for l in self.invariant_recombinants.keys():
            if not l == self.defining_locus:
                recs = self.invariant_recombinants[l]
                r = recs[0]
                s = s + "," + "-".join([r["V"], r["J"]])
                if len(recs) > 1:
                    for r in recs[1:]:
                        s = s + " | " + "-".join([r["V"], r["J"]])

        return s
