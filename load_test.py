import pickle

file = open('imdb.pkl', 'rb')
train_set = pickle.load(file)
test_set = pickle.load(file)

print(train_set[0][0])
print("========")
print(test_set[0][0])
