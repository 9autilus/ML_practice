import math
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import label_binarize

class MyMetric:
  mean_accuracy       = 0.0
  mean_f1_score       = 0.0
  mean_lift           = 0.0
  mean_roc_area       = 0.0
  mean_avg_precision  = 0.0
  mean_BEP            = 0.0
  mean_rmse           = 0.0
  mean_mxe            = 0.0
  
  mean_BEP_gap        = 0
  
  ################################################
  ################################################
 
  def __init__(self, init_val):
    mean_accuracy       = float(init_val)
    mean_f1_score       = float(init_val)
    mean_lift           = float(init_val)
    mean_roc_area       = float(init_val)
    mean_avg_precision  = float(init_val)
    mean_BEP            = float(init_val)
    mean_rmse           = float(init_val)
    mean_mxe            = float(init_val)

  ################################################
  ################################################
  def dual_column(self, y):
    temp = np.array(y)
    for i in xrange(temp.shape[0]):
      if temp[i] == 0:
        temp[i] = 1
      else:
        temp[i] = 0
    return np.concatenate((temp, y), 1)
  
  ################################################
  ################################################
  def compute_metrics(self, X_test, y_test, y_pred, y_score, classes_list):
    ## Number of classes
    n_classes = len(classes_list)

    ## In case of 2 classes, Convert the labels to binary {0,1}
    if n_classes == 2:
      y_test = label_binarize(y_test, classes_list)
      y_pred = label_binarize(y_pred, classes_list)
      y_test_bin = self.dual_column(y_test)
    else:
      y_test_bin = label_binarize(y_test, classes=classes_list)
    
    #---------------------------- Accuracy --------------------
    ## Mean accuracy (both give same results)
    #mean_accuracy = clf.score(X_test, y_test)            ## from probability values
    self.mean_accuracy = metrics.accuracy_score(y_test, y_pred) ##from predicted values

    #---------------------------- F1 Score --------------------
    ## F1-Score for each class
    '''
    print "Classes F1-score: "
    print metrics.f1_score(y_test, y_pred, average=None) ## Score each class separately. Returns array of size [n_classes]
    '''
    ## Mean F1-Score
    try:
      if n_classes == 2:
        self.mean_f1_score = (metrics.f1_score(y_test, y_pred, average='binary')) ## Weighted averaging
      else:
        self.mean_f1_score = (metrics.f1_score(y_test, y_pred, average='weighted')) ## Weighted averaging
    except ValueError:
      print "Mean F1-score: Error"

    #---------------------------- ROC Curve --------------------
    ##Compute ROC-curve for each class
    '''
    fpr = dict()
    tpr = dict()
    threshold = dict()
    for class_ctr in xrange(n_classes):
      fpr[class_ctr], tpr[class_ctr], threshold[class_ctr] = metrics.roc_curve(y_test_bin[:, class_ctr], y_score[:, class_ctr])
    '''

    #---------------------------- ROC Area --------------------
    ## ROC-area for each class
    '''
    roc_area = dict()
    for class_ctr in xrange(n_classes):
      roc_area[class_ctr] = metrics.roc_auc_score(y_test_bin[:, class_ctr], y_score[:, class_ctr])
      print "Class %d ROC-area: %5.3f" % (class_ctr, roc_area[class_ctr]) ## print each class
    '''
    
    ## Mean ROC-area (across all classes)
    try:
      self.mean_roc_area = (metrics.roc_auc_score(y_test_bin, y_score))
    except ValueError:
      print "Mean ROC-area: Error"

    #---------------------------- Average Precision --------------------
    ##Compute Average Precision for each class
    '''
    sum_precision = 0.0
    average_precision = dict()
    for class_ctr in xrange(n_classes):
      average_precision[class_ctr] = metrics.average_precision_score(y_test_bin[:, class_ctr], y_score[:, class_ctr])
      sum_precision += average_precision[class_ctr]
      print "Class %d Avg Precision: %5.3f" % (class_ctr, average_precision[class_ctr]) ## print each class
    '''
    ##Compute Mean Average Precision
    try:
      self.mean_avg_precision = (metrics.average_precision_score(y_test_bin, y_score))
    except ValueError:
      print "Mean Avg-precision: Error"

    #---------------------------- Precision Recall BEP --------------------
    ##Compute precision-recall Break Even Point (BEP) from precision-recall curve for each class
    precision   = dict()
    recall      = dict()
    threshold   = dict()
    BEP         = dict()
    sum_BEP     = 0.0
    sum_min_gap = 0.0

    for class_ctr in xrange(n_classes):
      min_gap = float("inf")
      BEP[class_ctr] = -1
      
      try:
        precision[class_ctr], recall[class_ctr], threshold[class_ctr] = \
          metrics.precision_recall_curve(y_test_bin[:, class_ctr],y_score[:, class_ctr])
      except ValueError:
        print "Error in BEP for class %d. Setting to 0" %class_ctr
        BEP[class_ctr] = 0
        continue
      
      for i in xrange(precision[class_ctr].shape[0]):
        if abs(precision[class_ctr][i] - recall[class_ctr][i]) < min_gap:
          min_gap = abs(precision[class_ctr][i] - recall[class_ctr][i])
          BEP[class_ctr] = precision[class_ctr][i]
      sum_BEP += BEP[class_ctr]
      sum_min_gap += min_gap
      #print "Class %d BEP: %5.3f with gap: %5.3f" % (class_ctr, BEP[class_ctr], min_gap) ## print each class
    self.mean_BEP     = sum_BEP/float(n_classes)
    self.mean_BEP_gap = sum_min_gap/float(n_classes)

    #---------------------------- RMSE --------------------
    try:
      self.mean_rmse = math.sqrt(metrics.mean_squared_error(y_test_bin, y_score))
    except ValueError:
      print "Root MSE: Error"


    #---------------------------- Cross-Entropy --------------------

    
  ################################################
  ################################################
  def mean_metric_from_kfold(self, metric_list):
    num_kfolds = len(metric_list)
    
    ## Sum all k-folds
    for i in metric_list:
      self.mean_accuracy      += i.mean_accuracy     
      self.mean_f1_score      += i.mean_f1_score     
      self.mean_lift          += i.mean_lift         
      self.mean_roc_area      += i.mean_roc_area     
      self.mean_avg_precision += i.mean_avg_precision
      self.mean_BEP           += i.mean_BEP          
      self.mean_rmse          += i.mean_rmse         
      self.mean_mxe           += i.mean_mxe          

      self.mean_BEP_gap       += i.mean_BEP_gap
      
    ## Divide by number of k-folds
    self.mean_accuracy      = self.mean_accuracy      / num_kfolds   
    self.mean_f1_score      = self.mean_f1_score      / num_kfolds
    self.mean_lift          = self.mean_lift          / num_kfolds
    self.mean_roc_area      = self.mean_roc_area      / num_kfolds
    self.mean_avg_precision = self.mean_avg_precision / num_kfolds
    self.mean_BEP           = self.mean_BEP           / num_kfolds
    self.mean_rmse          = self.mean_rmse          / num_kfolds
    self.mean_mxe           = self.mean_mxe           / num_kfolds
    
    self.mean_BEP_gap       = self.mean_BEP_gap       / num_kfolds
        
  
  ################################################
  ################################################
  def display_elaborate(self):
    print "Mean Accuracy      : %5.3f" % self.mean_accuracy
    print "Mean F1-score      : %5.3f" % self.mean_f1_score
    print "Mean ROC-area      : %5.3f" % self.mean_roc_area
    print "Mean Avg-precision : %5.3f" % self.mean_avg_precision
    print "Mean BEP           : %5.3f (with mean-gap: %5.3f)" % (self.mean_BEP, self.mean_BEP_gap)
    print "Root MSE           : %5.3f" % self.mean_rmse
  
  ################################################
  ################################################
  def display_in_a_row(self):
    print "[%5.3f %5.3f %5.3f    %5.3f %5.3f %5.3f    %5.3f %5.3f]" % (\
      self.mean_accuracy, self.mean_f1_score, self.mean_lift, \
      self.mean_roc_area, self.mean_avg_precision, self.mean_BEP, \
      self.mean_rmse, self.mean_mxe)
      