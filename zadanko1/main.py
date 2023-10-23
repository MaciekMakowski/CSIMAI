from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Wczytanie danych
data = load_breast_cancer()
X = data.data
y = data.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Inicjowanie klasyfikatora KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Możesz dostosować liczbę sąsiadów (n_neighbors) według potrzeb

# Trenowanie modelu na danych treningowych
knn.fit(X_train, y_train)

# Prognozowanie etykiet na zbiorze testowym
y_pred = knn.predict(X_test)

true_positives = sum((y_pred == 1) & (y_test == 1))
false_positives = sum((y_pred == 1) & (y_test == 0))
true_negatives = sum((y_pred == 0) & (y_test == 0))
false_negatives = sum((y_pred == 0) & (y_test == 1))
# Obliczanie precyzji
precision = true_positives / (true_positives + false_positives)

# Obliczanie czułości (sensitivity)
recall = true_positives / (true_positives + false_negatives)

# Obliczanie specyficzności (specificity)
specificity = true_negatives / (true_negatives + false_positives)

#Ujemna wartość przewidywana
negPredVal = true_negatives / (false_negatives + true_negatives)

# Obliczanie dokładności
accuracy = (true_positives + true_negatives) / ( true_positives + true_negatives + false_positives +false_negatives)

print(f'Precyzja: {precision}')
print(f'Czułość: {recall}')
print(f'Specyficzność: {specificity}')
print(f'Ujemna wartość przewidywana: {negPredVal}')
print(f'Dokładność: {accuracy}')