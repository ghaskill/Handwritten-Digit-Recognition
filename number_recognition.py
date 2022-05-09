import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

digits = load_digits()
(X_train, X_test, y_train, y_test) = train_test_split(
  digits.data, digits.target, test_size=0.3, random_state=11
)

ks = np.arange(2, 10)
scores = []
for k in ks:
  model = KNeighborsClassifier(n_neighbors=k)
  score = cross_val_score(model, X_train, y_train)
  score.mean()
  scores.append(score.mean())

final_model = KNeighborsClassifier(n_neighbors=3)
final_model.fit(digits.data, digits.target)

plt.plot(ks, scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

filename = 'knn_model_pickle.sav'
pickle.dump(final_model, open(filename, 'wb'))
