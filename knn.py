import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

import parse_dataset as parser
from metrics import MyMetric


## Setting to print array fully
np.set_printoptions(threshold=sys.maxsize)

datasets                = ['car', 'cancer', 'forest', 'nursery','optdigits'] ## Add more if need be
test_ratios             = [0.5, 0.9] ## Ratio of test set in full data set
cross_validation_ratio  = 0.3 ## Ratio of validation set in full training data
n_folds                 = 5
k_list                  = [1, 2, 3, 4, 5, 6, 10, 20, 30]
limit                   = 0 ## Number of feature vectors to read from file. 0: All of them
split_seed              = None

## For testing purpose. Override.
#datasets                = ['car', 'nursery', 'cancer', 'forest']
#test_ratios = [0.5] ## Ratio of test set in full data set
#n_folds = 1
#k_list                  = [1, 2, 3, 4, 5, 6, 10, 20, 30]
#limit = 0
#split_seed = 0

'''
Performs a k-fold cross validation on given training set for a given k in knn.
Divides the training set into [training, validation] based on the 
value of cross_validation_ratio parameter. 
Returns mean of each metric obtained from k-fold cross validation
'''
def k_fold_cross_validation(X, y, k_value, n_classes):
  k_fold_metric_list = []

  ## Loop over all k-folds
  for fold in xrange(n_folds):
    # Split into training and test
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=cross_validation_ratio, random_state = split_seed, stratify=y)

    num_tr_char = X_train.shape[0]
    num_cv_char = X_validation.shape[0]

    #print "X_train\n", X_train, "\n"
    #print "y_train\n", y_train, "\n"
    #print "X_test\n", X_test, "\n"
    #print "y_test\n", y_test, "\n\n"
    
    ####################### Training ##################
    if num_tr_char == 0:
      print "Insufficient training vectors: %d." % num_tr_char
      exit(0)

    #print "Traning started... ",
    clf = KNeighborsClassifier(n_neighbors=k_value)
    clf.fit(X_train,y_train)
    #print clf
    #print "SVM Training complete with %d samples." % num_tr_char

    ####################### Cross validation ##################
    if num_cv_char == 0:
      print "Insufficient testing vectors: %d." % num_cv_char
      exit(0)

    ## get all predicted scores from machine
    #print "Predicting digits...",
    #y_score = clf.decision_function(X_test)
    y_score = clf.predict_proba(X_validation)
    y_pred  = clf.predict(X_validation)
    #print "Done.\n"
    #print y_test
    #print y_pred
    #print (y_score)

    ######### Store Metrics for current train/test division #######
    metric = MyMetric(0)
    metric.compute_metrics(X_validation, y_validation, y_pred, y_score, n_classes)
    k_fold_metric_list.append(metric)
    
  ## Get the mean value of all metrics
  meanMetric = MyMetric(0.0)
  meanMetric.get_mean_metric(k_fold_metric_list) #Get average of all k-folds
      
  ## Return meanMetric
  return meanMetric

'''
Based on the input [X_train, y_train] this function the 
best value of parameter k
Returns the best k
'''
def get_best_parameters(X_train, y_train, n_classes):
  k_metric_list = []
  
  ## Loop over all values of k-neighbors to get error value for each
  for k_value in k_list:
    ## K-fold cross-validation
    metric = k_fold_cross_validation(X_train, y_train, k_value, n_classes)
    k_metric_list.append(metric)

    ## Display for current k_value in knn
    print "k= %d Metrics: " % (k_value),
    metric.display_in_a_row()
    
  ## Loop over all values of k-neighbors to find the best 
  ## Selecting based on accuracy as of now
  idx = 0
  performance = float("-inf")
  
  for i in xrange(len(k_metric_list)):
    if k_metric_list[i].mean_accuracy > performance:
      idx         = i
      performance = k_metric_list[idx].mean_accuracy
  
  ## Return the k which was found to have the best performance 
  ## in k-fold validation set
  return k_list[idx]

'''
Based on the input [best_c] this function first trains the machine 
on input [X_train, y_train]. 
And then tests it on input [X_test, y_test].
Returns the metrics for this test.
'''
def final_train_and_test(X_train, X_test, y_train, y_test, best_k, n_classes):

  clf = KNeighborsClassifier(n_neighbors=best_k)
  clf.fit(X_train,y_train)
  
  y_score = clf.predict_proba(X_test)
  y_pred  = clf.predict(X_test)
  
  best_metric = MyMetric(0)
  best_metric.compute_metrics(X_test, y_test, y_pred, y_score, n_classes)  
  
  return best_metric
  

'''
For a given test_size_val, this function divides the input X and y
into two parts. Training and Testing.
Training set is then fed to get_best_parameters() to find the best 
parameters for the machine.
One the best parameters are known, the machine is applied on the Test
set. Metrics are populated and displayed.
'''
def process_a_ratio(X, y, test_size_val, n_classes):
  ## Get a ratio
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state = split_seed, stratify=y)
  
  best_k = get_best_parameters(X_train, y_train, n_classes)

  ## Use this value of k in knn and compute final results
  best_metric = final_train_and_test(X_train, X_test, y_train, y_test, best_k, n_classes)
  
  ## Report the results
  print "Results for Test-Ratio %3.2f " % test_size_val
  print "k= %d Metrics: " % (best_k),
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



