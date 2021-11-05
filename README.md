# ml-ids
Code used to build an intrusion detection system using machine learning. In this work the dataset that will be used is NSL-KDD Data Set, a subset of the KDD Cup 99 Data Set that solves some of the problems in this. More informations about this dataset can be found in https://www.unb.ca/cic/datasets/nsl.html.


The dataset.py file contains the code used to perform the analysis of the dataset.

The folder figures contains the figures generated by the file explained above as part of the dataset study.

The folder data contains the files with the train set, test set and related files from the NSL-KDD Data Set.

The folder stats contains text files generated by dataset.py with information about the attributes with non-numeric values. For each possible value of the attributes 'service' and 'protocol' we can find a list with the following information: number of samples that contain that value and represent an attack, percentage that these attacks represent above the total of attacks, percentage that have this value and are attacks above the total samples that contain that attribute value, percentage that are not attacks (with that value) with respect to the total of samples with that value.


