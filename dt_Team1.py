from preprocessing import df_input
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Seeding with Group Number
sklearn.random.seed(1)

# split dataset into features and target variables
feature_cols = ['sex=female', 'sex=male','race=white','race=asian-pac-islander','race=black','race=amer-indian-eskimo','race=other',
                'capital_gain', 'capital_loss', 'country', 'young', 'adult', 'senior', 'old', 'part_time', 'full_time', 
                'overtime', 'gov', 'Not-Working', 'Private', 'Self Employed', 'Married', 'Never-married', 'Not Married',
                'Exec-managerial', 'Prof-specialty', 'Other', 'ManualWork', 'Sales']

target_cols = ['income'] # Target attribute

X = df_input[feature_cols]
Y = df_input[target_cols]
test_size = 0.3 # 70% train 30% test


# Split the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1) 


# CRITERION HYPERPARAMETER

# Create Decision Tree Classifier Object
gini_classifier = DecisionTreeClassifier(criterion="gini", random_state=1)
entropy_classifier = DecisionTreeClassifier(criterion="entropy", random_state=1,)

# Train the Decision Tree Classifiers
gini_classifier = gini_classifier.fit(X_train, Y_train)
entropy_classifier = entropy_classifier.fit(X_train, Y_train)

# Predict the response for the dataset
Y_pred_gini = gini_classifier.predict(X_test)
Y_pred_entropy = entropy_classifier.predict(X_test)

print("\nDecision Tree Metrics:\n")
print("Gini Accuracy: ", round(metrics.accuracy_score(Y_test, Y_pred_gini), 2))
print("Entropy Accuracy: ", round(metrics.accuracy_score(Y_test, Y_pred_entropy), 2))


# MIN_LEAF_SIZE HYPERPARAMETER

# Create Decision Tree Classifier Object
min_5_leaf_classifier = DecisionTreeClassifier(min_samples_leaf=5, random_state=1)
min_10_leaf_classifier = DecisionTreeClassifier(min_samples_leaf=10, random_state=1)
min_15_leaf_classifier = DecisionTreeClassifier(min_samples_leaf=15, random_state=1)


# Train the Decision Tree Classifiers
min_5_leaf_classifier = min_5_leaf_classifier.fit(X_train, Y_train)
min_10_leaf_classifier = min_10_leaf_classifier.fit(X_train, Y_train)
min_15_leaf_classifier = min_15_leaf_classifier.fit(X_train, Y_train)


# Predict the response for the dataset
Y_pred_min_5_leaf = min_5_leaf_classifier.predict(X_test)
Y_pred_min_10_leaf = min_10_leaf_classifier.predict(X_test)
Y_pred_min_15_leaf = min_15_leaf_classifier.predict(X_test)


print("\n\n\nMin 5 Leaf Accuracy: ", round(metrics.accuracy_score(Y_test, Y_pred_min_5_leaf), 2))
print("\nModel Metrics:\n", metrics.classification_report(Y_test, Y_pred_min_5_leaf))

print("\nMin 10 Leaf Accuracy: ", round(metrics.accuracy_score(Y_test, Y_pred_min_10_leaf), 2))
print("\nModel Metrics:\n", metrics.classification_report(Y_test, Y_pred_min_10_leaf))

print("\nMin 15 Leaf Accuracy: ", round(metrics.accuracy_score(Y_test, Y_pred_min_15_leaf), 2))
print("\nModel Metrics:\n", metrics.classification_report(Y_test, Y_pred_min_15_leaf))


fileName = "pred_dt_1.csv"

with open(fileName, 'w') as f:
    f.write('\n'.join(Y_pred_min_5_leaf))
