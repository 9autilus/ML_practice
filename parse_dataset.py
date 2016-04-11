

def parse_optdigits_file(f, file_tag, limit=0):
  print "Parsing file: ", file_tag
  
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
     
    X.append(feature_list)    
    y.append(label)
          
    ## Increment the number of characters read
    feature_ctr = feature_ctr + 1
    print "\rParsing vector #: ", feature_ctr, 
    
    ## Decrement the iteration counter
    if limit:
      cond = cond - 1
    
  print "Done Parsing file: ", file_tag
  return (X, y)  

  
def parse_forest_file(f, file_tag, limit=0):
  print "Parsing file: ", file_tag

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
    
    ## Pop the label data from feature_list
    label = feature_list.pop(0)
    #print "label: ", label
    
    #if (label < 0) or (label > 9):
    #  print "Parsing error in tranining. label %d found\n" %label
    #  break
     
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
    
  print "Done Parsing file: ", file_tag
  return (X, y)
  

def parse_cancer_file(f, file_tag, limit=0):
  print "Parsing file: ", file_tag

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
    
    ## feature list is in string format right now. Convert it to int
    feature_list = [int(i) for i in feature_list]
    #print feature_list
    
    ## Test the validity of labels
    if (label != 2) and (label != 4):
      #print "Parsing error in tranining. label %d found\n" %label
      continue
      
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
  
  print "Done Parsing file: ", file_tag
  return (X, y)