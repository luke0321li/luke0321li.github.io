---
title: A very quick note on the confusion matrix 
---
![placeholder](https://chloenelkin.files.wordpress.com/2012/12/cat-7-mondrian-composition-c-no-iii-with-red-yellow-and-blue-1935_stretchcmyk-no-6-in-photo-sheet1.jpg)
*Composition C by Piet Mondiran.*

## Motivation
The confusion matrix plays a huge role in evaluating performance of a statistical model (or a diagnostic device). Below is a self-explanatory example of a binary confusion matrix.

![placeholder](https://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix_files/confusion_matrix_1.png)

For example, assume someone developed a device that detects ADHD. He then conducted some clinical tests to justify the validity of the device. It is given that, of all the 1000 test subjects, 200 actually has ADHD and the rest are control. The device, oblivious of the actual distribution, identified 180 ADHD-positive individuals. Among these 180 subjects, 170 are "real" ADHD patients and the rest are controls misclassified by the device. In this case, we have 170 true positives (TPs), 180 - 170 = 10 false positives (FPs), 800 - 10 = 790 true negatives (TNs), and 200 - 170 = 30 false negatives (FNs).

## Metrics
Natually, one would want as many TP and TN as possible relative to FP and FN. Therefore, scientists developed many derived values from the confusion matrix. Some of them are used interchangeably and are indeed the same thing. Some of them are used ambiguously but are in fact not the same. Here is a list of all the metrics derived from the confusion matrix and their usage:

- **Precision** (with no alternative names so far) is $$\frac{TP}{TP + FP}$$ in other words how many samples identified as positive by the model/device are actually positive. It is a measure of how well the model avoids classifying negative samples as positive. 

- **Recall**, also called **Sensitivity** or **True Positive Rate**, is $$\frac{TP}{TP + FN}$$. Since the sum of TP and FN is the total number of positive samples, recall measures how well the model "picks out" the positive samples, i.e. what proportion of all the positive samples are correctly identified. 

- **Specificity**, also called **True Negative Rate**, is $$\frac{TN}{TN + FP}$$. It is like recall but for negative samples. The sum of TN and FP is the total number of negative samples. Therefore, specificity measures what proportion of all the negative samples are correctly identified. 

People usually say "precision & recall" or "sensitivity & specificity" as if they are intrinsically paired. Indeed, precision and recall are both used to compute another metric called F1-score. Higher F1-score indicates better accuracy because it is the harmonic mean of precision ($$P$$) and recall ($$R$$):

$$F_{1} = 2\frac{PR}{P + R}$$

On the other hand, sensitivity ($$TPR$$) and 1 - specificity ($$1 - TNR = FPR$$, false positive rate) are the y- and x-axis for the ROC (receiver operating characteristic) curve respectively. There will probably be another article discussing the ROC in detail coming soon. 

## Final notes
A multitide of statistical metrics have been invented to address the vagueness of the term "model accuracy". Also, having high score for one metric does not necessarily imply the same for another. A model may have great sensitivity but poor precision because it simply identifies all samples as positive.    
