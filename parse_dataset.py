

filenames         = dict()
class_labels      = dict()
parsing_function  = dict()


def parse_cancer_file(f, label_dictionary, limit=0):
  X = []
  y = []
  feature_ctr = 0
  cond = 1
  
  if limit:
    cond = limit  
  
  while(cond):
    ## Import the first {label, feature}
    feature_string = f.readline()
    if len(feature_string) == 0:
      #print "Warning: Ending training. Short bmp %d found\n" %len(bmp_string)
      break;
    # Discard the last newline character if present
    if feature_string[-1] == '\n':
      feature_string = feature_string[:-1]
      
    feature_list = feature_string.split(',')
    #print "feature_string (splitted): ", feature_list
    
    ## Discard the first column with ID
    feature_list = feature_list[1:]
    
    ## Pop the label data from feature_list
    label = int(feature_list.pop(-1))
    #print "label: ", label
    
    ## Test the validity of features. Discard features-vectors with missing features
    missing_feature = False
    for i in feature_list:
      if i == '?':
        missing_feature = True
    if missing_feature: #Discard this feature-vector
      continue
    
    ## ignore any blank lines in the file
    if len(feature_list) < 2:
      continue    
    
    ## feature list is in string format right now. Convert it to int
    feature_list = [int(i) for i in feature_list]
    #print feature_list
    
    ## Test the validity of labels
    if (label != 2) and (label != 4):
      #print "Parsing error in tranining. label %d found\n" %label
      continue
      
    ## Convert labels to a numbers: 0 to n_classes - 1)
    label = label_dictionary[label]      
      
    ## Test the validity of features. Feature values should be from 1 to 10.
    for i in feature_list:
      if i < 0 or i > 10:
        #print "Parsing error in tranining. label %d found\n" %label
        continue

    X.append(feature_list)    
    y.append(label)

    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing training vector #: ", feature_ctr,
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1     
  
  print "Done Parsing file."
  return (X, y)
  

def parse_forest_file(f, label_dictionary, limit=0):
  X = []
  y = []
  feature_ctr = 0
  cond = 1
  
  if limit:
    cond = limit  
  
  #Discard the first dummy line from the file
  f.readline()

  while(cond):
    ## Import the first {label, feature}
    feature_string = f.readline()
    if len(feature_string) == 0:
      #print "Warning: Ending training. Short bmp %d found\n" %len(bmp_string)
      break;
    # Discard the last newline character if present
    if feature_string[-1] == '\n':
      feature_string = feature_string[:-1]
      
    feature_list = feature_string.split(',')
    #print "feature_string (splitted): ", feature_list
    
    ## ignore any blank lines in the file
    if len(feature_list) < 2:
      continue    
    
    ## Pop the label data from feature_list
    ## The [0] index is added to get rid of the whitespace after class label char in file
    label = feature_list.pop(0)[0]
    #print "label: ", label
    
    #if (label < 0) or (label > 9):
    #  print "Parsing error in tranining. label %d found\n" %label
    #  break
     
    ## Convert labels to a numbers: 0 to n_classes - 1)
    label = label_dictionary[label]     
     
    ## feature list is in string format right now. Convert it to float
    feature_list = [float(i) for i in feature_list]
    #print feature_list
     
    X.append(feature_list)    
    y.append(label)

    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing training vector #: ", feature_ctr,
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1     
    
  print "Done Parsing file."
  return (X, y)
  
  
def parse_optdigits_file(f, label_dictionary, limit=0):
  char_wd = 32
  char_ht = 32
  bmp_wd = char_wd + 1
  bmp_ht = char_ht  
  
  X = []
  y = []
  feature_ctr = 0
  cond = 1
  
  if limit:
    cond = limit
  
  #Discard the first 21 dummy lines from the file
  for i in xrange(21):
    f.readline()

  while(cond):
  #for i in xrange(0,1):
    ## Import the BMP data
    feature_string = f.read(bmp_wd * bmp_ht)
    if len(feature_string) != bmp_wd * bmp_ht:
      if(len(feature_string) != 0):
        print "Warning: Ending training. Short bmp %d found\n" %len(feature_string)
      break;
      
    feature_list = []
    for line in feature_string.split('\n'):
      feature_list = feature_list + list(line)
    # print "feature_list of len: ", len(feature_list), "\n", feature_list, "\n"
    
    ## Import the label data
    label = int(f.readline(bmp_wd))
    if (label < 0) or (label > 9):
      print "Parsing error in tranining. label %d found\n" %label
      break  
     
    ## Convert labels to a numbers: 0 to n_classes - 1)
    label = label_dictionary[label]
     
    X.append(feature_list)    
    y.append(label)
          
    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing vector #: ", feature_ctr, 
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1
    
  print "Done Parsing file."
  return (X, y)  

  
def parse_car_file(f, label_dictionary, limit=0):
  X = []
  y = []
  feature_ctr = 0
  cond = 1
  
  ## Feature vectors
  buying    = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
  maint     = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
  doors     = {'2': 0, '3': 1, '4': 2, '5more': 3}
  persons   = {'2': 0, '4': 1, 'more': 2}
  lug_boot  = {'small': 0, 'med': 1, 'big': 2}
  safety    = {'low': 0, 'med': 1, 'high': 2}
  
  if limit:
    cond = limit  
  
  while(cond):
    ## Import the first {label, feature}
    feature_string = f.readline()
    if len(feature_string) == 0:
      #print "Warning: Ending training. Short bmp %d found\n" %len(bmp_string)
      break;
    # Discard the last newline character if present
    if feature_string[-1] == '\n':
      feature_string = feature_string[:-1]
      
    feature_list = feature_string.split(',')
    #print "feature_string (splitted): ", feature_list
    
    ## ignore any blank lines in the file
    if len(feature_list) < 2:
      continue    
    
    ## Pop the label data from feature_list
    ## The [0] index is added to get rid of the whitespace after class label char in file
    label = feature_list.pop(-1)
    #print "label: ", label
    
    ## Convert labels to a numbers: 0 to n_classes - 1)
    label = label_dictionary[label]
    
    #if (label < 0) or (label > 9):
    #  print "Parsing error in tranining. label %d found\n" %label
    #  break
     
    #print feature_list
    
    feature_list[0] = buying[feature_list[0]]
    feature_list[1] = maint[feature_list[1]]
    feature_list[2] = doors[feature_list[2]]
    feature_list[3] = persons[feature_list[3]] 
    feature_list[4] = lug_boot[feature_list[4]]
    feature_list[5] = safety[feature_list[5]]
    #print feature_list
    
    X.append(feature_list)    
    y.append(label)

    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing training vector #: ", feature_ctr,
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1     
    
  print "Done Parsing file."
  #from collections import Counter 
  #print Counter(y) ## Print how many feature-vectors each class have  
  return (X, y)
  
  
def parse_nursery_file(f, label_dictionary, limit=0):
  X = []
  y = []
  feature_ctr = 0
  cond = 1
  
  parents   = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
  has_nurs  = {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4}
  form      = {'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3}
  children  = {'1': 0, '2': 1, '3': 2, 'more': 3}
  housing   = {'convenient': 0, 'less_conv': 1, 'critical': 2}
  finance   = {'convenient': 0, 'inconv': 1}
  social    = {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2}
  health    = {'recommended': 0, 'priority': 1, 'not_recom': 2}
  
  if limit:
    cond = limit  
  
  while(cond):
    ## Import the first {label, feature}
    feature_string = f.readline()
    if len(feature_string) == 0:
      #print "Warning: Ending training. Short bmp %d found\n" %len(bmp_string)
      break;
    # Discard the last newline character if present
    if feature_string[-1] == '\n':
      feature_string = feature_string[:-1]
      
    feature_list = feature_string.split(',')
    #print "feature_string (splitted): ", feature_list
    
    ## ignore any blank lines in the file
    if len(feature_list) < 2:
      continue
    
    ## Pop the label data from feature_list
    ## The [0] index is added to get rid of the whitespace after class label char in file
    label = feature_list.pop(-1)
    #print "label: ", label
    
    ## Convert labels to a numbers: 0 to n_classes - 1)
    label = label_dictionary[label]    
    
    #if (label < 0) or (label > 9):
    #  print "Parsing error in tranining. label %d found\n" %label
    #  break
     
    #print feature_list
    
    feature_list[0] = parents[feature_list[0]]
    feature_list[1] = has_nurs[feature_list[1]]
    feature_list[2] = form[feature_list[2]]
    feature_list[3] = children[feature_list[3]] 
    feature_list[4] = housing[feature_list[4]]
    feature_list[5] = finance[feature_list[5]]
    feature_list[6] = social[feature_list[6]]
    feature_list[7] = health[feature_list[7]]
    #print feature_list
    
    X.append(feature_list)    
    y.append(label)

    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing training vector #: ", feature_ctr,
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1     
    
  print "Done Parsing file."
  # from collections import Counter 
  # print Counter(y) ## Print how many feature-vectors each class have
  return (X, y)
    
  
def init_parser():
  filenames['nursery']    = r"E:\Course\CAP6610 Machine Learning\Project\dataset\nursery\nursery.data"
  filenames['car']        = r"E:\Course\CAP6610 Machine Learning\Project\dataset\car\car.data"
  filenames['cancer']     = r"E:\Course\CAP6610 Machine Learning\Project\dataset\BreastCancer\breast-cancer-wisconsin.data"
  filenames['forest']     = r"E:\Course\CAP6610 Machine Learning\Project\dataset\ForestType\combined.csv"
  filenames['optdigits']  = r"E:\Course\CAP6610 Machine Learning\Project\dataset\optdigits\optdigits-orig-combined.tra"

  ## Assign IDs to each label
  ## It's utmost import that all class labels be assigned an ID from 0 to (n_classes - 1)
  class_labels['nursery']    = {'not_recom':0, 'very_recom':1, 'priority':2, 'spec_prior':3}
  class_labels['car']        = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}
  class_labels['cancer']     = {2:0, 4:1}
  class_labels['forest']     = {'d':0, 'h':1, 'o':2, 's':3}
  class_labels['optdigits']  = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6 ,7:7, 8:8, 9:9}

  parsing_function['nursery']   = parse_nursery_file
  parsing_function['car']       = parse_car_file
  parsing_function['cancer']    = parse_cancer_file
  parsing_function['forest']    = parse_forest_file
  parsing_function['optdigits'] = parse_optdigits_file

'''
Returns:
X: Feature vectors
y: Class labels (Not raw labels read from file. It returns label-ID, which
   is a number from 0 to (n_classes - 1)
last param: The third parameter is the number of classes present in test data
'''  
def parse_a_dataset(dataset_key, limit = 0):
  f = open(filenames[dataset_key], "r")
  
  X, y = parsing_function[dataset_key](f, class_labels[dataset_key], limit)
  
  return X, y, len(class_labels[dataset_key])
  
  

  
  
  
  