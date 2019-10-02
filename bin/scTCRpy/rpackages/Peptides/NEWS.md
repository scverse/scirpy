NEWS
====
**Peptides v.2.3**

* aaComp function now accepts character lists as input. Thanks to Hemílio Xafranski from their suggestion.

**Peptides v.2.3**

* A problem with the tScales and tsScales functions was solved. The problem made the return of the functions an empty list. Thanks to Fabian Tann <fabian.tann@lse.thm.de> to report it.

**Peptides v.2.2**

* The Wimley-White hydrophobicity scales were added thanks to Alexander Komin <akomin1@jhu.edu> suggestion. WW scales can be used as:
1. **interfaceScale_pH2:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25
2. **interfaceScale_pH8:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25
3. **octanolScale_pH2:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25
4. **octanolScale_pH8:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25
5. **oiScale_pH2:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25
6. **oiScale_pH8:** White, Stephen (2006-06-29). "Experimentally Determined Hydrophobicity Scales". University of California, Irvine. Retrieved 2017-05-25


**Peptides v.2.1**

* The charge and pI functions were rewritten in C++ and an optimization approach was used thanks to Scott McCain (@jspmccain) and Timothée Poisot (@tpoisot) suggestion.

* An error in zScales function data was solved. Q and E values were wrongly interchanged in v 2.0.

**Peptides v.2.0.0**

* All datasets were unified into AAdata

* All test were migrated to testthat

* readXVG and plotXVG functions were improved by J. Sebastian Paez

* kideraFactors output vector was renamed as KF#

* Now all sequences are checked before to property calculation

* aaDescriptos, fasgaiVectors, blosumIndices, mswhimScores, zScales, vhseScales, protFP, tScales and stScales functions were added

**Peptides v.1.2.2**

* crucianiProperties function was added.

**Peptides v.1.2.1**

* Four new functions were added: autoCorrelation, autoCovariance, crossCovariance and crucianiProperties

* Functions related with XVG files were updated.

* Documentation was changed to roxygen2

**Peptides v.1.1.2**

* All functions were re-vectorized to support set of peptides as input

* Kidera function now returns all factors in a unique output

**Peptides v.1.1.1**

* The mw function now computes the molecular weight using monoisotopic values

* A problem with blank spaces was solved
 
**Peptides v.1.1.0**

* The kidera function and Kfactors dataset was included.

**Peptides v.1.0.4**

* A instaindex function bug has been fixed.

* A problem with line breaks in sequences was solved.

**Peptides v.1.0.3**
* A membpos function bug has been fixed.

* The results now are not rounded.

**Peptides v.1.0.2**

* Hydrophobicity function now can compute the GRAVY index with one of the 38 scales includes in Peptides (*new):

  1. **Aboderin:** Aboderin, A. A. (1971). An empirical hydrophobicity scale for α-amino-acids and some of its applications. International Journal of Biochemistry, 2(11), 537-544.
  2. **AbrahamLeo:** Abraham D.J., Leo A.J. Hydrophobicity (delta G1/2 cal). Proteins: Structure, Function and Genetics 2:130-152(1987).
  3. ***Argos:** Argos, P., Rao, J. K., & Hargrave, P. A. (1982). Structural Prediction of Membrane‐Bound Proteins. European Journal of Biochemistry, 128(2‐3), 565-575.
  4. **BlackMould:** Black S.D., Mould D.R. Hydrophobicity of physiological L-alpha amino acids. Anal. Biochem. 193:72-82(1991).
  5. **BullBreese:** Bull H.B., Breese K. Hydrophobicity (free energy of transfer to surface in kcal/mole). Arch. Biochem. Biophys. 161:665-670(1974).
  6. ***Casari:** Casari, G., & Sippl, M. J. (1992). Structure-derived hydrophobic potential: hydrophobic potential derived from X-ray structures of globular proteins is able to identify native folds. Journal of molecular biology, 224(3), 725-732.
  7. **Chothia:** Chothia, C. (1976). The nature of the accessible and buried surfaces in proteins. Journal of molecular biology, 105(1), 1-12.
  8. ***Cid:** Cid, H., Bunster, M., Canales, M., & Gazitúa, F. (1992). Hydrophobicity and structural classes in proteins. Protein engineering, 5(5), 373-375.
  9. **Cowan3.4:** Cowan R., Whittaker R.G. Hydrophobicity indices at pH 3.4 determined by HPLC. Peptide Research 3:75-80(1990).
  10. **Cowan7.5:** Cowan R., Whittaker R.G. Hydrophobicity indices at pH 7.5 determined by HPLC. Peptide Research 3:75-80(1990).
  11. **Eisenberg:** Eisenberg D., Schwarz E., Komarony M., Wall R. Normalized consensus hydrophobicity scale. J. Mol. Biol. 179:125-142(1984).
  12. ***Engelman:** Engelman, D. M., Steitz, T. A., & Goldman, A. (1986). Identifying nonpolar transbilayer helices in amino acid sequences of membrane proteins. Annual review of biophysics and biophysical chemistry, 15(1), 321-353.
  13. ***Fasman:** Fasman, G. D. (Ed.). (1989). Prediction of protein structure and the principles of protein conformation. Springer.
  14. **Fauchere:** Fauchere J.-L., Pliska V.E. Hydrophobicity scale (pi-r). Eur. J. Med. Chem. 18:369-375(1983).
  15. ***Goldsack:** Goldsack, D. E., & Chalifoux, R. C. (1973). Contribution of the free energy of mixing of hydrophobic side chains to the stability of the tertiary structure of proteins. Journal of theoretical biology, 39(3), 645-651.
  16. **Guy:** Guy H.R. Hydrophobicity scale based on free energy of transfer (kcal/mole). Biophys J. 47:61-70(1985).
  17. **HoppWoods:** Hopp T.P., Woods K.R. Hydrophilicity. Proc. Natl. Acad. Sci. U.S.A. 78:3824-3828(1981).
  18. **Janin:** Janin J. Free energy of transfer from inside to outside of a globular protein. Nature 277:491-492(1979).
  19. ***Jones:** Jones, D. D. (1975). Amino acid properties and side-chain orientation in proteins: a cross correlation approach. Journal of theoretical biology, 50(1), 167-183.
  20. ***Juretic:** Juretic, D., Lucic, B., Zucic, D., & Trinajstic, N. (1998). Protein transmembrane structure: recognition and prediction by using hydrophobicity scales through preference functions. Theoretical and computational chemistry, 5, 405-445.
  21. ***Kidera:** Kidera, A., Konishi, Y., Oka, M., Ooi, T., & Scheraga, H. A. (1985). Statistical analysis of the physical properties of the 20 naturally occurring amino acids. Journal of Protein Chemistry, 4(1), 23-55.
  22. ***Kuhn:** Kuhn, L. A., Swanson, C. A., Pique, M. E., Tainer, J. A., & Getzoff, E. D. (1995). Atomic and residue hydrophilicity in the context of folded protein structures. Proteins: Structure, Function, and Bioinformatics, 23(4), 536-547.
  23. **KyteDoolittle:** Kyte J., Doolittle R.F. Hydropathicity. J. Mol. Biol. 157:105-132(1982).
  24. ***Levitt:** Levitt, M. (1976). A simplified representation of protein conformations for rapid simulation of protein folding. Journal of molecular biology, 104(1), 59-107.
  25. **Manavalan:** Manavalan P., Ponnuswamy Average surrounding hydrophobicity. P.K. Nature 275:673-674(1978).
  26. **Miyazawa:** Miyazawa S., Jernigen R.L. Hydrophobicity scale (contact energy derived from 3D data). Macromolecules 18:534-552(1985).
  27. **Parker:** Parker J.M.R., Guo D., Hodges R.S. Hydrophilicity scale derived from HPLC peptide retention times. Biochemistry 25:5425-5431(1986).
  28. ***Ponnuswamy:** Ponnuswamy, P. K. (1993). Hydrophobic charactesristics of folded proteins. Progress in biophysics and molecular biology, 59(1), 57-103.
  29. ***Prabhakaran:** Prabhakaran, M. (1990). The distribution of physical, chemical and conformational properties in signal and nascent peptides. Biochem. J, 269, 691-696.
  30. **Rao:** Rao M.J.K., Argos P. Membrane buried helix parameter. Biochim. Biophys. Acta 869:197-214(1986).
  31. **Rose:** Rose G.D., Geselowitz A.R., Lesser G.J., Lee R.H., Zehfus M.H. Mean fractional area loss (f) [average area buried/standard state area]. Science 229:834-838(1985)
  32. **Roseman:** Roseman M.A. Hydrophobicity scale (pi-r). J. Mol. Biol. 200:513-522(1988).
  33. **Sweet:** Sweet R.M., Eisenberg D. Optimized matching hydrophobicity (OMH). J. Mol. Biol. 171:479-488(1983).
  34. **Tanford:** Tanford C. Hydrophobicity scale (Contribution of hydrophobic interactions to the stability of the globular conformation of proteins). J. Am. Chem. Soc. 84:4240-4274(1962).
  35. **Welling:** Welling G.W., Weijer W.J., Van der Zee R., Welling-Wester S. Antigenicity value X 10. FEBS Lett. 188:215-218(1985).
  36. **Wilson:** Wilson K.J., Honegger A., Stotzel R.P., Hughes G.J. Hydrophobic constants derived from HPLC peptide retention times. Biochem. J. 199:31-41(1981).
  37. **Wolfenden:** Wolfenden R.V., Andersson L., Cullis P.M., Southgate C.C.F. Hydration potential (kcal/mole) at 25C. Biochemistry 20:849-855(1981).
  38. ***Zimmerman:** Zimmerman, J. M., Eliezer, N., & Simha, R. (1968). The characterization of amino acid sequences in proteins by statistical methods. Journal of theoretical biology, 21(2), 170-201.


* The mw function has been fixed to give the same result as ExPASy pI/mw tool.
* The hmoment function is now vectorized and allow adjust the windows size. (thanks to an anonymous reviewer of RJournal).
* The pepdata dataset and the variable name are now unified to lowercases.
* The seqinr package dependency was removed.
