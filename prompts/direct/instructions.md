You will receive the title and the abstract of a PubMed article as JSON with the fields "title" and "abstract".

Mark every chemical entity mention in both fields by wrapping it in <chem> and </chem>, following the guidelines below.

Return JSON with the same two fields. In each field, copy the original text exactly, character for character, and only insert the tags. Do not change, add, or remove any other character, including spaces and punctuation. Do not nest tags. If a field contains no chemical entity mention, return it unchanged.

{guidelines}
