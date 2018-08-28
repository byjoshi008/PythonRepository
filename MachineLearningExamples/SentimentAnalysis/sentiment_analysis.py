from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

with open("./data/amazon_cells_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')
with open("./data/yelp_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')
with open("./data/imdb_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')

lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']

train_documents = [line[0] for line in lines]
train_labels = [line[1] for line in lines]

count_vectorizer = CountVectorizer(binary='true')
train_documents = count_vectorizer.fit_transform(train_documents)

classifier = BernoulliNB().fit(train_documents, train_labels)

prediction = classifier.predict(count_vectorizer.transform(["The movie was fine though I did not like unnecessary action sequences."]))

print("Negative" if prediction == '0' else "Positive")
