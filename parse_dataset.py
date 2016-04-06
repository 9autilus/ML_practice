




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
