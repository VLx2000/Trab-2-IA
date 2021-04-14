from sklearn.datasets import load_wine
# X - Matriz de atributos y - vetor de classes
X, y = load_wine(return_X_y=True)

from sklearn.preprocessing import OneHotEncoder
OneHotEncoder(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn import tree
arvore = tree.DecisionTreeClassifier(criterion='entropy')

arvore = arvore.fit(X_train, y_train)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10,10))
fig = tree.plot_tree(arvore)

previsto = arvore.predict(X_test)

from sklearn.metrics import plot_confusion_matrix
class_names = y
title = "Matriz Confusão"
disp = plot_confusion_matrix(arvore, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
disp.ax_.set_title(title)
plt.show()

from sklearn.metrics import classification_report
print("Relatório")
print(f"{classification_report(y_test, previsto)}")