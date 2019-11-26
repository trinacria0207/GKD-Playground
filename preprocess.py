import pandas as pd
import numpy as np
import re
import json

train = pd.read_csv("train.csv")
train_data = [] 
label = []
#print(train['question1'][0])
#n = len(train['question1'])
n = 10000
#print(train_data.shape)
train_words = []
train_ = []
max_len = 0
for i in range(n):
#    train_data.append([train['question1'][i],train['question2'][i]])
    sen1 = re.split('\W',train['question1'][i])
    sen2 = re.split('\W',train['question2'][i])
    train_.append([sen1, sen2])
    label.append(int(train['is_duplicate'][i]))
    train_words += sen1+sen2
    max_len = max(max_len, len(sen1), len(sen2))
    
vocab= set(train_words)
word2id = {w: i+1 for i, w in enumerate(list(vocab))}
word2id['UNK'] = 0
# pad = 0

data_numpy = np.zeros((len(train_), max_len, 2))
for i in range(len(train_)):
    for j in range(2):
        for k, word in enumerate(train_[i][j]):
            data_numpy[i,k,j] = word2id[word]
print(data_numpy.shape)
data_label = np.array(label)
train_data = data_numpy[:int(n*0.8)]
test_data = data_numpy[int(n*0.8):]

train_label = data_label[:int(n*0.8)]
test_label = data_label[int(n*0.8):]

np.save('train_data', train_data)
np.save('train_label', train_label)
np.save('test_data', test_data)
np.save('test_label', test_label)

with open('vocab.json', 'w') as fp:
    json.dump(word2id,fp)



#train_data = []
#for word in train_words:
#    train_data.append(word2id[word])

#print(train_sentences)
#print(np.shape(train_words))
#for i in range(list(vocab)):
#    word2id[i] = vocab
#print(word2id)

# voc file
# train_data numpy file (N,2,ws)
# tabel (N,1) numpy file 


#    train_data[i].append(train['question2'][i])
#train_data=np.asarray(train_data)
#label = np.asarray(label)

#print(train_data.shape)
#print(label.shape)


## Word to id
## voc size 
#print(train_data[0][0])
