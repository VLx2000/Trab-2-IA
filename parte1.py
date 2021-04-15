from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# X - Matriz de atributos y - vetor de classes
X, y = load_wine(return_X_y=True)

# Transformando nominais em binarios
OneHotEncoder(X)

# Separar em conjunto treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

arvore = tree.DecisionTreeClassifier(criterion='entropy')

# algoritmo de induçao
arvore = arvore.fit(X_train, y_train)

# Plota arvore
fig = plt.figure(figsize=(10,10))
fig = tree.plot_tree(arvore)

# Classifica os dados
previsto = arvore.predict(X_test)

# Plota Matriz
class_names = y
title = "Matriz Confusão"
disp = plot_confusion_matrix(arvore, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
disp.ax_.set_title(title)
plt.show()

# Printa resultados
print("Relatório")
print(f"{classification_report(y_test, previsto)}")