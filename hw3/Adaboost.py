import sys
import math
t=int(sys.argv[1])
trainfile=sys.argv[2]
testfile=sys.argv[3]

trainingset=[]
#load training examples
with open(trainfile) as f:
 for line in f:
  l=line.rstrip().split()
  trainingset.append(l)

testset=[]
#load testset
with open(testfile) as f:
 for line in f:
  #print 'line='+line
  l=line.rstrip().split()
  #print 'len(l)='+str(len(l))
  testset.append(l)


num_features=len(trainingset[0])-1

#function to calculate overall entropy
def entropy():
 dpositive=0
 dnegative=0
 #print trainingset
 #print len(trainingset)
 #print D
 for i in range(len(trainingset)):
  #print trainingset[i][0]
  if trainingset[i][0]=='e':
   dpositive+=D[i]
  else:
   dnegative+=D[i]
 #print dpositive
 #print dnegative
 p1=dpositive/(dpositive+dnegative)
 p2=dnegative/(dpositive+dnegative)
 #print p1
 #print p2
 math.log(p1,2)
 math.log(p2,2)
 ans=(p1*math.log(p1,2))+(p2*math.log(p2,2))
 #print -1*ans
 return -1*ans

values={x:[] for x in range(1,len(trainingset[0]))}
#populate possible values taken up by different features
for l in trainingset:
 for i in range(1,len(trainingset[0])):
  if l[i] not in values[i]:
   values[i].append(l[i])
  
#function to calculate entropy for particlar value of feature
def entropy_feature_val(index, val):
 #print 'val='+val
 dpositive=0
 dnegative=0
 for i in range(len(trainingset)):
  if trainingset[i][index]==val:
   if trainingset[i][0]=='e':
    dpositive+=D[i]
   else:
    dnegative+=D[i]
 #print dpositive
 #print dnegative
 p1=dpositive/(dpositive+dnegative)
 p2=dnegative/(dpositive+dnegative)
 if p1==0:
  ans=p2*math.log(p2,2)
  if p2==0:
   ans=0
 elif p2==0:
  ans=p1*math.log(p1,2)
 else:
  ans=((p1)*math.log(p1,2))+((p2)*math.log(p2,2))
 return -1*ans

   



#function to calculate feature entropy
def entropy_feature(index):
 entropy_values={}
 for v in values[index]:
  entropy_values[v]=entropy_feature_val(index,v)
 return sum(entropy_values.values())
 



#function to return splitting attribute index
def getsplitting():
 tot_entropy=entropy()
 candidate_gain=[]
 for i in range(num_features):
  index=1+i #feature index is 1+ ith feature 
  #get entropy for the feature
  feature_entropy=entropy_feature(index)
  candidate_gain.append(tot_entropy-feature_entropy)
 return 1+candidate_gain.index(max(candidate_gain)) 
  

#initial D
D=[1/float(len(trainingset))]*len(trainingset)


#map e to 1 and p to -1
results_map={'e':1,'p':-1}


#store decision stumps
decision_stumps=[]
alpha=[]
for i in range(t):
 #select splitting attribute
 attindex=getsplitting() 
 #figure out the e,p distribution for different feature values
 leaf_values={v:[0,0] for v in values[attindex]}
 for l in trainingset:
  if l[0]=='e':
   leaf_values[l[attindex]][0]+=1
  else:
   leaf_values[l[attindex]][0]+=1
 #decide leaf values 
 decision_dict={} #keys=feature values, value=decision (e/p)
 for v in values[attindex]:
  if leaf_values[v][0]>leaf_values[v][1]:
   decision_dict[v]='e'
  else:
   decision_dict[v]='p'
 decision_stumps.append({attindex:decision_dict})
 #classify all training examples using this decision stump for alpha, epsilon
 epsilon=0
 training_results=[]
 for i in range(len(trainingset)):
  verdict=decision_dict[trainingset[i][attindex]]
  training_results.append(verdict)
  if verdict!=trainingset[i][0]:
   epsilon+=D[i]
 #print 'epsilon='+str(epsilon)
 temp_alpha= 0.5*math.log((1-epsilon)/epsilon, math.exp(1))
 alpha.append(temp_alpha)
 
 #calculate dnext
 dnext=[D[i]*math.exp(-1*temp_alpha*results_map[trainingset[i][0]]*results_map[training_results[i]]) for i in range(len(D))] 
 #normalize dnext
 dnext=[x/sum(dnext) for x in dnext]
 D=dnext


for x in alpha:
 print x


#now classify testset
#store answers
answers=[l[0] for l in testset]
test_results=[]
for l in testset:
 sum_result=0
 for i in range(t):
  splitting_attribute_i=decision_stumps[i].keys()[0]
  feature_value=l[splitting_attribute_i]
  result=results_map[decision_stumps[i][splitting_attribute_i][feature_value]]
  sum_result+=alpha[i]*result
 if(sum_result>=0):
  test_results.append('e')
 else:
  test_results.append('p')

print answers
print test_results
correct=0
for i in range(len(answers)):
 if answers[i]==test_results[i]:
  correct+=1
print correct
print len(answers)
print 'accuracy='+str(correct/float(len(answers)))
