import cPickle

file = open('imdb.pkl', 'rb')
train_set = cPickle.load(file)
test_set = cPickle.load(file)

print(train_set[0][0])
print("========")
print(test_set[0][0])
