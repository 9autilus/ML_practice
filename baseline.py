import sys
import numpy as np
from sklearn import dummy, preprocessing
from sklearn.cross_validation import train_test_split

import parse_dataset as parser
from metrics import MyMetric


## Setting to print array fully
np.set_printoptions(threshold=sys.maxsize)

datasets                = ['car', 'nursery', 'cancer', 'forest', 'optdigits'] ## Add more if need be
test_ratios             = [0.5, 0.9] ## Ratio of test set in full data set
limit                   = 0 ## Number of feature vectors to read from file. 0: All of them
split_seed              = None

## Overriding in order to limit the computation while debugging the code
#datasets                = ['cancer']#'car']#, 'nursery', 'cancer', 'forest']
#test_ratios = [0.5] ## Ratio of test set in full data set
#limit = 0
#split_seed = 0

'''
For a given test_size_val, this function divides the input X and y
into two parts. Training and Testing.
Training set is then fed to classifier.
Metrics are populated and displayed.
'''
def process_a_ratio(X, y, test_size_val, n_classes):
  ## Get a ratio
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state = split_seed, stratify=y)
  
  clf = dummy.DummyClassifier(strategy='most_frequent')
  clf.fit(X_train,y_train)
  
  y_score = clf.predict_proba(X_test)
  y_pred  = clf.predict(X_test)
  
  #from collections import Counter 
  #print Counter(y_train) ## Print how many feature-vectors each class have  
  #print Counter(y_test) ## Print how many feature-vectors each class have  
  #print Counter(y_pred) ## Print how many feature-vectors each class have  
  
  ## Adding a dummy label for each class so that there is atleast
  ## one instance of each label in predicted results. This is needed 
  ## because without it my metrics function is throwing error.
  ## This addition needn't be done in X_test because X_test is 
  ## not used anywhere after it.
  length = y_pred.shape[0]
  label_list = []
  for i in xrange(n_classes):
    label_list.append(i)
    score_list = np.array([0] * n_classes)
    score_list[i] = 1
    y_score = np.vstack([y_score, score_list])
  
  y_pred = np.hstack([y_pred, label_list])
  y_test = np.hstack([y_test, label_list])
  
  best_metric = MyMetric(0)
  best_metric.compute_metrics(X_test, y_test, y_pred, y_score, n_classes)    
  
  ## Report the results
  print "Results for Test-Ratio %3.2f " % test_size_val
  print "Baseline! Metrics: ",
  best_metric.display_in_a_row()  

'''
Main function.
Loops over all datasets.
Then loops over different ration settings. Each ratio setting gives
the results in the form of metrics.
'''
if __name__ == "__main__":
  ## Initiallize parser
  parser.init_parser()

  ## Loop over all datasets
  for dataset in datasets:
    print "\n\n########## Dataset: ", dataset, "##########"
    ## Read full data from current dataset
    X, y, n_classes = parser.parse_a_dataset(dataset, limit)
    
    X = np.array(X)
    y = np.array(y)
    
    ## Normalization if need be
    if dataset == 'forest':
      X = preprocessing.scale(X)
      #X = preprocessing.normalize(X, norm='l2')

    ## Loop over all the ratios. And see how they affect the results
    for test_size_val in test_ratios:
      print "\n\n------------------ Test size: %3.2f -----------------" % test_size_val
      process_a_ratio(X, y, test_size_val, n_classes)




