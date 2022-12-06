# SBN-Process
Code for Preprocessing SBN data for graph neural models and seq2seq models.
### Steps
* Download raw data from PMB website

* Combine individual files into one complete file based on different languages
  
  not all the text file has corresponding SBN file, check it twice in code.

* Split the data (train, dev, test)
  
  need to change path and language in file according to official document (README-PMB))

* Get the SBN data format for Seq2Seq models

* Get the SBN data format for Graph2Seq models
### Run


```
sh run-process.sh
```
