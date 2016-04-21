import math
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import label_binarize

## To calculate lift
from collections import Counter 

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
  ## n_classes (in the parameters) is the number of classes present in test data
  def compute_metrics(self, X_test, y_test, y_pred, y_score, n_classes):
    ## class label IDs
    classes_list = range(n_classes)
    
    ## In case of 2 classes, Convert the labels to binary {0,1}
    if n_classes == 2:
      y_test = label_binarize(y_test, classes_list)
      y_pred = label_binarize(y_pred, classes_list)
      y_test_bin = self.dual_column(y_test)
      y_pred_bin = self.dual_column(y_pred)
    else:
      y_test_bin = label_binarize(y_test, classes=classes_list)
      y_pred_bin = label_binarize(y_pred, classes=classes_list)
    
    #print "y_test:\n", y_test    
    #print "y_pred:\n", y_pred    
    #print "y_test_bin:\n", y_test_bin
    #print "y_score:\n", y_score    
    
    #---------------------------- Accuracy --------------------
    ## Accuracy for each class
    '''
    for class_ctr in xrange(n_classes):
      print "Class %d Accuracy: %5.3f" % (class_ctr, metrics.accuracy_score(\
        y_test_bin[:, class_ctr], \
        label_binarize(y_pred, classes=classes_list)[:, class_ctr])) ## print each class
    '''
    #print "y_test",     y_test
    #print "y_pred",     y_pred
    #print "y_test_bin", y_test_bin
    
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

    #---------------------------- Lift --------------------
    #print Counter(y) ## Print how many feature-vectors each class have  
    num_patterns    = y_test.shape[0]
    lift_threshold  = int(num_patterns/4) ## 25% of test vectors
    lift_sum        = 0
    
    if lift_threshold > 0 and (num_patterns - lift_threshold) > 1:
      #print "lift_threshold: ", lift_threshold, "num_patterns: ", num_patterns
      
      for class_ctr in xrange(n_classes):
        sorted_score_index = np.argsort(y_score[:, class_ctr])[::-1]
        
        a = 0; b = 0; c = 0; d = 0;
        
        for i in xrange(lift_threshold):
          if y_pred_bin[sorted_score_index[i]][class_ctr] == 1:
            a = a + 1
          else:
            c = c + 1
        
        for i in xrange(lift_threshold, num_patterns):
          if y_pred_bin[sorted_score_index[i]][class_ctr] == 1:
            b = b + 1
          else:
            d = d + 1      
      
        try:
          class_lift_numerator   = (a)/float(a + b)
          class_lift_denominator = (a + c)/float(a + b + c + d)
          
          class_lift = class_lift_numerator/class_lift_denominator
        except:
          ## It's not an error. 
          ## It's a limitation that if data set is too small, lift cannot be calculated.
          ## Since the division is not defined. Signal nan in lift value.
          class_lift = float('nan')
          #print "lift warning: Class %d a,b,c,d: %d %d %d %d" % (class_ctr, a,b,c,d)
        
        #print "Class %d lift: %5.3f" % (class_ctr, class_lift)
        lift_sum += class_lift
        
      self.mean_lift = lift_sum/float(n_classes)    
    
    '''
    num_pattern_lift = int(y_test.shape[0]/4) ## 25% of test vectors
    lift_sum = 0
    
    if num_pattern_lift > 1:
      #print "num_pattern_lift: ", num_pattern_lift
      
      for class_ctr in xrange(n_classes):
        sorted_score_index = np.argsort(y_score[:, class_ctr])[::-1][0: num_pattern_lift]
        
        a = 0; b = 0; c = 0; d = 0;
        #print "sorted_score_index:", sorted_score_index
        
        for i in xrange(len(sorted_score_index)):
          if y_pred_bin[sorted_score_index[i]][class_ctr] == 1:
            if y_test_bin[sorted_score_index[i]][class_ctr] == 1:
              a = a + 1
            else:
              c = c + 1
          else:
            if y_test_bin[sorted_score_index[i]][class_ctr] == 1:
              b = b + 1
            else:
              d = d + 1            
        
        try:
          class_lift_numerator   = (a)/float(a + b)
          class_lift_denominator = (a + c)/float(a + b + c + d)
          
          class_lift = class_lift_numerator/class_lift_denominator
        except:
          ## It's not an error. 
          ## It's a limitation that if data set is too small, lift cannot be calculated.
          ## Since the division is not defined. Signal nan in lift value.
          class_lift = float('nan')
          #print "lift warning: Class %d a,b,c,d: %d %d %d %d" % (class_ctr, a,b,c,d)
        
        #print "Class %d lift: %5.3f" % (class_ctr, class_lift)
        lift_sum += class_lift
        
      self.mean_lift = lift_sum/float(n_classes)
      '''
      
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
    ## Verification for RMSE is still pending
    ## I don't even know if (in multiclass case) this is the average of RMSE of all classes
    try:
      self.mean_rmse = math.sqrt(metrics.mean_squared_error(y_test_bin, y_score))
    except ValueError:
      print "Root MSE: Error"
    
    #---------------------------- Cross-Entropy --------------------
    try:
      ## Verification for Cross-Entropy is still pending
      ## I don't even know if (in multiclass case) this is the average of MXE of all classes
      ## First augument can be y_test or y_test_bin. It doesn't matter.
      self.mean_mxe = metrics.log_loss(y_test, y_score)
    except:
      print "Cross Entropy: Error"
    
  ################################################
  ################################################
  def get_mean_metric(self, metric_list):
    total_count = len(metric_list)
    
    ## Sum all
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
      
    ## Divide by total_count
    self.mean_accuracy      = self.mean_accuracy      / total_count   
    self.mean_f1_score      = self.mean_f1_score      / total_count
    self.mean_lift          = self.mean_lift          / total_count
    self.mean_roc_area      = self.mean_roc_area      / total_count
    self.mean_avg_precision = self.mean_avg_precision / total_count
    self.mean_BEP           = self.mean_BEP           / total_count
    self.mean_rmse          = self.mean_rmse          / total_count
    self.mean_mxe           = self.mean_mxe           / total_count
    
    self.mean_BEP_gap       = self.mean_BEP_gap       / total_count
        
  
  ################################################
  ################################################
  def display_elaborate(self):
    print "Mean Accuracy      : %5.3f" % self.mean_accuracy
    print "Mean F1-score      : %5.3f" % self.mean_f1_score
    print "Mean ROC-area      : %5.3f" % self.mean_roc_area
    print "Mean Avg-precision : %5.3f" % self.mean_avg_precision
    print "Mean BEP           : %5.3f (with mean-gap: %5.3f)" % (self.mean_BEP, self.mean_BEP_gap)
    print "Root MSE           : %5.3f" % self.mean_rmse
    print "Cross-Entropy MXE  : %5.3f" % self.mean_rmse
  
  ################################################
  ################################################
  def display_in_a_row(self):
    print "[%5.3f %5.3f %5.3f    %5.3f %5.3f %5.3f    %5.3f %5.3f]" % (\
      self.mean_accuracy, self.mean_f1_score, self.mean_lift, \
      self.mean_roc_area, self.mean_avg_precision, self.mean_BEP, \
      self.mean_rmse, self.mean_mxe)
      