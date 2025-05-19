import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('drug.csv')

missing_values = data.isnull().sum()

print("Number of missing values")
print(missing_values)

data = data.dropna()

encodedData = pd.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

X = encodedData.drop('Drug', axis=1)
y = encodedData['Drug']
Accuracies = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train, y_train)
    yPred = DTC.predict(X_test)
    accuracy = accuracy_score(y_test, yPred)
    Accuracies.append(accuracy)
    print(f"Experiment {i + 1}:Test Set Size: {len(X_test)}, Accuracy: {accuracy}")

MaxAccuracy = max(Accuracies)
BestModel = Accuracies.index(MaxAccuracy) + 1
print(f"Best Model is Experiment {BestModel} with Accuracy: {MaxAccuracy}")

# second experiment
mean_accuracy = []
max_accuracy = []
min_accuracy = []
mean_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []

train_sizes = np.arange(0.3, 0.8, 0.1)

for train_size in train_sizes:
    mean_acc_list = []
    max_acc_list = []
    min_acc_list = []
    mean_tree_size_list = []
    max_tree_size_list = []
    min_tree_size_list = []

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=i)
        DTC = DecisionTreeClassifier()
        DTC.fit(X_train, y_train)
        y_pred = DTC.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        tree_size = DTC.tree_.node_count

        mean_acc_list.append(accuracy)
        max_acc_list.append(accuracy)
        min_acc_list.append(accuracy)

        mean_tree_size_list.append(tree_size)
        max_tree_size_list.append(tree_size)
        min_tree_size_list.append(tree_size)


    mean_accuracy.append(np.mean(mean_acc_list))
    max_accuracy.append(np.max(max_acc_list))
    min_accuracy.append(np.min(min_acc_list))

    mean_tree_sizes.append(np.mean(mean_tree_size_list))
    max_tree_sizes.append(np.max(max_tree_size_list))
    min_tree_sizes.append(np.min(min_tree_size_list))


print("\n Second Experiment")
for i, train_size in enumerate(train_sizes):
    print(f"\nTraining Set Size: {round(train_size * 100, 2)}%")
    print(
        f"Mean Accuracy: {round(mean_accuracy[i], 4)}, Max Accuracy: {round(max_accuracy[i], 4)}, Min Accuracy: {round(min_accuracy[i], 4)}")
    print(
        f"Mean Tree Size: {round(mean_tree_sizes[i], 2)}, Max Tree Size: {round(max_tree_sizes[i], 2)}, Min Tree Size: {round(min_tree_sizes[i], 2)}")

report_data = {
    'Training Set Size (%)': [round(size * 100, 2) for size in train_sizes],
    'Mean Accuracy': mean_accuracy,
    'Max Accuracy': max_accuracy,
    'Min Accuracy': min_accuracy,
    'Mean Tree Size': mean_tree_sizes,
    'Max Tree Size': max_tree_sizes,
    'Min Tree Size': min_tree_sizes
}

report_df = pd.DataFrame(report_data)
report_df.to_csv('experiment_report.csv', index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_sizes, mean_accuracy, label='Mean Accuracy')
plt.plot(train_sizes, max_accuracy, label='Max Accuracy', linestyle='dashed')
plt.plot(train_sizes, min_accuracy, label='Min Accuracy', linestyle='dashed')
plt.title('Accuracy vs Training Set Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_sizes, mean_tree_sizes, label='Mean Tree Size')
plt.plot(train_sizes, max_tree_sizes, label='Max Tree Size', linestyle='dashed')
plt.plot(train_sizes, min_tree_sizes, label='Min Tree Size', linestyle='dashed')
plt.title('Number of Nodes in Final Tree vs Training Set Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Number of Nodes')
plt.legend()

plt.tight_layout()
plt.show()


