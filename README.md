# 
# HMMF:
#### A Hybrid Multi-modal Fusion Framework for Predicting Drug Side Effect Frequencies

### Quick start
Data Directory
The data directory contains several files that are crucial for the functionality of HMMF. Here's an overview of each file:

1. drug_description.xlsx
Description: This file contains [Text description of drugs in benchmark data set].
2. side_description.xlsx
Description: [Text description of side effects in benchmark data set].
3. drug_side_frequencies.pkl
Description: [Frequency matrix of drug side effects].
4. semantic.pkl
Description: [Similarity of hyponymy in knowledge base(ADRecs)].
5. drug_drug_scores.pkl
Description: [Similarity of drugs collected in STITCH database].
6. drug_disease_jaccard.pkl
Description: [Correlation similarity of drug diseases collected in CTD database].
7. word.pkl
Description: [Wikipedia Pre-training Word Vector Similarity].

Make sure to review the contents of each file in the data directory for a better understanding of the input data used by HMMF. If you need more details on the data format or structure, refer to the documentation or comments within the source code.

Feel free to reach out if you have any questions or need further assistance with the HMMF framework.
- Run HMMF:
```
$ nohup python main.py 
```