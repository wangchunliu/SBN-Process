# SBN-Process
Code for Preprocessing SBN data for graph neural models and seq2seq models.
### Steps
1. Download raw data from PMB website

2. Combine individual files into one complete file based on different languages

3. Split the data (train, dev, test, need to change path and language in file according to official document (README-PMB))

4. Get the SBN data format for Seq2Seq models

5. Get the SBN data format for Graph2Seq models
### Run


```
sh run-process.sh

```
