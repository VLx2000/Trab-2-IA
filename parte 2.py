from sklearn.datasets import load_diabetes
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

X, y = load_diabetes(return_X_y=True)

OneHotEncoder(X)

soma_quadrados_distancias = []
group_range = range(1, 15)

for i in group_range:
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(X)

    soma_quadrados_distancias.append(kmeans.inertia_)

for j in range(1, 15):
    print(f"Grupo {j}: {soma_quadrados_distancias[j - 1]}")

plt.plot(group_range, soma_quadrados_distancias, 'bx-')
plt.xlabel('group range')
plt.ylabel('Soma dos Quadrados das Distâncias')
plt.title('Cotovelo')
plt.show()
