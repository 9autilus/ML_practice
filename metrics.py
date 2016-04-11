import math
from sklearn import svm, metrics

def compute_metrics(X_test, y_test, y_test_bin, y_pred, y_score):
  ## Number of classes
  n_classes = y_test_bin.shape[1]
  
  #---------------------------- Accuracy --------------------
  ## Mean accuracy (both give same results)
  #mean_accuracy = clf.score(X_test, y_test)            ## from probability values
  mean_accuracy = metrics.accuracy_score(y_test, y_pred) ##from predicted values
  print "Mean Accuracy: %5.3f" % mean_accuracy

  #---------------------------- F1 Score --------------------
  ## F1-Score for each class
  '''
  print "Classes F1-score: "
  print metrics.f1_score(y_test, y_pred, average=None) ## Score each class separately. Returns array of size [n_classes]
  '''
  ## Mean F1-Score
  print "Mean F1-score: %5.3f" % (metrics.f1_score(y_test, y_pred, average='weighted')) ## Weighted averaging

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
  print "Mean ROC-area: %5.3f" % (metrics.roc_auc_score(y_test_bin, y_score))

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
  print "Mean Avg-precision: %5.3f" % (metrics.average_precision_score(y_test_bin, y_score))

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
    
    precision[class_ctr], recall[class_ctr], threshold[class_ctr] = \
      metrics.precision_recall_curve(y_test_bin[:, class_ctr],y_score[:, class_ctr])
    
    for i in xrange(precision[class_ctr].shape[0]):
      if abs(precision[class_ctr][i] - recall[class_ctr][i]) < min_gap:
        min_gap = abs(precision[class_ctr][i] - recall[class_ctr][i])
        BEP[class_ctr] = precision[class_ctr][i]
    sum_BEP += BEP[class_ctr]
    sum_min_gap += min_gap
    #print "Class %d BEP: %5.3f with gap: %5.3f" % (class_ctr, BEP[class_ctr], min_gap) ## print each class
  print "Mean BEP: %5.3f with mean-gap: %5.3f" % (sum_BEP/float(n_classes), sum_min_gap/float(n_classes))

  #---------------------------- RMSE --------------------
  print "Root MSE: %5.3f" % math.sqrt(metrics.mean_squared_error(y_test_bin, y_score))


  #---------------------------- Cross-Entropy --------------------






  
    
    
    