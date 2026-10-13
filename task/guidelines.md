<!--
This task definition is shared by every condition of the experiment (the same text is given to all of them).
It summarizes the official CHEMDNER annotation guidelines:
  Rabal, O., Krallinger, M. "CHEMDNER data preparation and annotation manual", version 2.0 (31 July 2013).
  Zenodo record 4957519 (CEM_guidelines.pdf), licensed under CC BY 4.0. https://zenodo.org/records/4957519
The summary rewords the rules. Examples are included only where the manual states explicitly what is and is not tagged.
-->

# Task: chemical entity mention recognition (CHEMDNER CEM)

Find every mention of a chemical entity in the title and in the abstract of a PubMed article. A mention is a contiguous span of characters in the original text. Mark exactly the characters that belong to the chemical name.

## What counts as a chemical entity mention

A chemical entity mention (CEM) is a name of a specific chemical, a specific class of chemicals, or a fragment of a specific chemical, that can be linked to chemical structure information with some reliability. General chemical concepts, proteins, and macromolecular biochemicals are not CEMs.

Tag:

- Nouns that correspond to a single structure: atoms, ions, isotopes, elements, molecules (fluorine, iron, deuterium, benzene, pyridine).
- Structural classes, functional groups, and fragments that correspond to a generic structure (amides, hydroxypyridines, ROH, amino acids).
- General class names whose definition carries structural or compositional information, whether synthetic or natural (hydrocarbons, organochlorines, carbohydrates, steroids, terpenoids, fatty acids, nucleotides, monosaccharides, transition metal, halogen, Schiff base).
- Small biochemicals: mono-, di-, and trisaccharides (glucose, sucrose, streptomycin); peptides with fewer than 15 amino acids, including their chemical modifications (glutathione, cyclosporin A, GnRH, angiotensin); nucleotide monomers, dimers, and trimers (NADH, NAD+, ATP, cAMP, S-adenosyl methionine); lipids such as fatty acids and their derivatives and sterol derivatives, excluding polymeric structures (glycerol, prostaglandin A, cholesterol).
- Synthetic polymers (nylon, polystyrene, polyvinyl chloride).
- Minerals (calcite, silica, alumina), laboratory reagents with a well-defined composition (petroleum ether, silica gel), and dyes and indicators (methyl red, Coomassie Brilliant blue, DAPI).
- Database identifiers of chemicals (CAS registry numbers, PubChem, ChEBI, ChEMBL) and company codes, when the context shows they refer to a chemical.
- Formulae, SMILES, InChI, and three-letter codes of amino acids, nucleotides, and monosaccharides (C6H12O6, EtOAc, Fe2+, D-Ala-D-Ala, GlcNAc).
- Trivial, trade, and generic drug names (aspirin, paracetamol, Viagra), and abbreviations or acronyms of chemicals that can be clearly linked to a chemical (DMSO, GABA).

Do not tag:

- Words that are not chemical nouns: isolated adjectives, pronouns and anaphors ("this compound"), verbs, chemical reactions (methylation, hydrolysis), enzymes, and isolated chemical prefixes ("1,4-" in "1,4-derivatives").
- Compound numbers, even next to a name: in "8-amino-2,6-methano-3-benzazocine (2)" do not tag "2".
- Words naming a role rather than a structure: analogue, inhibitor, agonist, antioxidant, drug, compound, hormone, antibiotic, toxin, metabolite, solvent, reagent, buffer, catalyst, electrophile, cation, anion.
- Very non-specific structural concepts: atom, ion, molecule, polymer, isomer, enantiomer, monomer, dimer, lipid, protein, functional group, aglycone, oligosaccharide, glycoside.
- Words that only look like chemicals in context ("lead compounds", "an iron will").
- Proteins and polypeptides with more than 15 amino acids, nucleic acid polymers, polysaccharides, and other large biopolymers (insulin, DNA, mRNA, collagen, starch, cellulose, glycogen, peptidoglycans, glycoproteins).
- Vague mixtures (melanin).
- By convention: "organic", "inorganic", water in any state and the adjective "aqueous", "proton", "lead", and "gold". "Iron" is tagged.
- Unclear cases: if you are not sure that a word is a chemical, do not tag it.

## Where a mention starts and ends

- Tag the longest name, but only the words that belong to the chemical name. Leave out non-essential parts and modifiers: in "sodium ion", "methyl group", "carbon atom", "caffeine analogue", "cocaine addiction", "pyridine derivatives", tag only "sodium", "methyl", "carbon", "caffeine", "cocaine", "pyridine".
- Keep a word such as "substituted" when it is inside a chemical name ("N-substituted-2-alkyl-3-hydroxy-4(1H)-pyridinones"), but not when it stands alone and gives no specific information ("disubstituted naphthalenes", "14-substituted derivatives of carminomycinone": leave out "disubstituted" and "14-substituted").
- "Acid", "base", "salt", and "metal" are tagged only as part of a longer specific name or as explicit classes ("lysergic acid", "carboxylic acid", "transition metal", "citric acid trisodium salt"). Alone they are not tagged ("in treatment with aqueous alkali or acid": tag neither "alkali" nor "acid"). The exception is "salt" meaning sodium chloride.
- An adjective is included only when it is next to a valid chemical name and adds structural information; the adjective and the noun are tagged together as one mention ("polychlorinated biphenyl", "acetylated phenols", "ethanolic KOH", "allylic alcohol", "primary alcohols", "aqueous sodium carbonate"). In "ionotropic glutamate receptors", "ionotropic" is not tagged.
- An adjective next to a general class word (compounds, analogues, derivatives, series) is tagged alone when it adds specific structural information ("pyrazolic" in "pyrazolic compounds"). Adjectives that leave a very wide family are not tagged ("methoxylated analogues", "fluorinated compounds": nothing is tagged). The prefix "non-" is left out ("non-steroidal": tag only "steroidal").
- Do not include flanking spaces, commas, full stops, brackets, or citation markers. Keep square brackets that are part of the name of an inorganic complex or an ionic liquid.
- Include prefixes for stereochemistry or regiochemistry ("cis-platin", "(S)-alanine"), stoichiometry numbers in formulae ("C6H8O3.2H2O"), and oxidation-state, charge, state, and surface symbols at the end of a name ("Cu2+", "Cu(II)", "CuSO4(aq)", "(14)C").
- Position numbers are included only when they are part of the name. In "C-2 carbon" tag only "carbon"; in "C-2 and C-3 positions" tag nothing; in "Ser473" tag only "Ser". A position that identifies a general position in a compound class is included (the "4-" of "4-bromo" in "4-bromo derivative"), for numeric and for named positions (ortho, meta, para, o-).
- Do not include trademark symbols ("Aspirin®": tag "Aspirin"), trailing hyphens, or a possessive apostrophe-s ("benzene's": tag "benzene"). Keep a trailing hyphen in CAS-style names and family mentions.
- Do not break up a word to reach a chemical inside it ("methylating", "acetyltransferase", "mGluR2": tag nothing).
- A chemical inside an enzyme name is tagged ("pyruvate" in "pyruvate kinase", "K+" in "K+ ATPase", "phosphatidylinositol" in "phosphatidylinositol 3-kinase"). When the chemical and the enzyme are joined by a hyphen, split them and analyze each part ("pyruvate" in "pyruvate-kinase", "K+" in "K+-ATPase"). For "A B -ase" enzymes (an enzyme converting A to B), tag A and B separately ("squalene" and "hopene" in "squalene hopene cyclase").
- A chemical inside another kind of expression is tagged ("1H" in "1H NMR", "Na+" in "d[Na+]/dt", "caffeine" in "[caffeine]=10 mM").
- Misspelled names and text-conversion errors are tagged. Names in other languages are tagged.
- A mention never spans two sentences.
- Do not tag acronyms of one or two letters, one-letter codes of amino acids or nucleotides, or mutation mentions ("Arg176Met", "A385V"). One-letter symbols of chemical elements are tagged ("N" meaning nitrogen; "Pd" and "C" in "Pd/C").

## Lists and multi-word names

- When full names are listed, tag each one separately ("citric acid and acetic acid": two mentions; "lithium carbonate, sodium carbonate": two mentions).
- When names share a part and are not written out in full, tag the whole string, including "and", "or", and commas, as one mention ("citric and acetic acid", "lithium, sodium and potassium carbonate", "thieno2,3-d and thieno3,2-d fused oxazin-4-ones").
- If words inside such a phrase are not part of the chemical name, split it ("elaidic (t-C18:1 delta9) and palmitic acid" contains two mentions).
- Two adjacent chemical words that denote one entity form one mention ("adenine nucleotide", "alkyl acetates", "isopropyl halides", "carbonyl carbon", "methyl ether", "pyrimidine nucleosides", "limonoid glycosides"). Mixtures written "CEM1/CEM2" or "CEM1-CEM2" are separate mentions ("hexane-ethyl acetate": "hexane" and "ethyl acetate"; "Pd/C": "Pd" and "C"). If other words separate the two names, tag them separately.
- A name given together with its abbreviation or definition is tagged as separate mentions ("gamma-aminobutyric acid (GABA)": "gamma-aminobutyric acid" and "GABA").
