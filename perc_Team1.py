# grid search total epochs for the perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from preprocessing import df_input
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

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

# Split the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=1) # 70% train 30% test

#Train the scaler 
sc = StandardScaler()
sc.fit(X_train)

#Apply the scaler to the X training data 
X_train_std = sc.transform(X_train)

#Apply the scaler to the X test 
X_test_std = sc.transform(X_test)

ppn1 = Perceptron(max_iter=40, eta0=0.2, random_state=2)
# Train the perceptron
ppn1.fit(X_train_std, Y_train)

#Apply the trained perceptron on the X data to make predicts for the Y test data 
Y_pred1 = ppn1.predict(X_test_std)


# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
print('\n\nAccuracy: %.9f' % accuracy_score(Y_test, Y_pred1))

print(classification_report(Y_test, Y_pred1))

#Testing hyperparameters eta0
ppn2 = Perceptron(max_iter=26, eta0=0.2, random_state=1) 
# Train the perceptron
ppn2.fit(X_train_std, Y_train)

#Apply the trained perceptron on the X data to make predicts for the Y test data 
Y_pred2 = ppn2.predict(X_test_std)

# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
print('\n\nAccuracy: %.9f' % accuracy_score(Y_test, Y_pred2))

print(classification_report(Y_test, Y_pred2))

#Testing hyperparameters eta0
ppn3 = Perceptron(max_iter=11, eta0=0.2, random_state=5) 
# Train the perceptron
ppn3.fit(X_train_std, Y_train)

#Apply the trained perceptron on the X data to make predicts for the Y test data 
Y_pred3 = ppn3.predict(X_test_std)

# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
print('\n\nAccuracy: %.9f' % accuracy_score(Y_test, Y_pred3))

print(classification_report(Y_test, Y_pred3))


print('\nWe choose the model that uses max_iter that has the higher max_iter because it produces better accuracy. \n')
fileName = 'pred_perc_1.csv'
with open(fileName, 'w') as f:
    f.write('\n'.join(Y_pred1))

