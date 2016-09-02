from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import sys
import fileinput

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


training_file = open("trainingdata.txt", "r")

line_list = list(training_file)
data = []
labels = []
target_names = [1,2,3,4,5,6,7,8]

qline_list = []

for line in sys.stdin:
    #print(line)
    qline_list.append(line)
qline_list.pop(0);

for line in line_list:
    line_partition = (line.partition(' '))
    #print(line_partition[0])
    labels.append(line_partition[0])
    #print(line_partition[2])
    data.append(line_partition[2])


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

text_clf2 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-2, n_iter=1, random_state=42)),
                     ])

text_clf = text_clf.fit(data, labels)

#for line in qline_list:
#    print(line)

#X_new_counts = count_vect.transform(qline_list)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = text_clf.predict(qline_list)
for label in predicted:
    print(label)