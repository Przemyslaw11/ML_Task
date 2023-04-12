# %%
import pandas as pd
import numpy as np

''' 1. Load the Covertype Data Set '''

df = pd.read_csv('covtype.data')

# %%
''' 2. Implement a very simple heuristic that will classify the data '''
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=42)

# Calculate the mean value of each feature for each class in the training set
class_means = []
for i in range(1,8):
    class_means.append(np.mean(X_train[y_train==i], axis=0))

# Define the model function
def heuristic_model(row):
    distances = [np.linalg.norm(row - mean) for mean in class_means]
    return np.argmin(distances)+1

# Test the model on the test data
y_pred = [heuristic_model(row) for index, row in X_test.iterrows()]

# Calculate the accuracy of the classification
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the model as a pickle file
with open('heur_model.pkl', 'wb') as f:
    pickle.dump(heuristic_model, f)

# %%
''' 3. Use Scikit-learn library to train two simple Machine Learning models '''
''' Decision Tree Classifier: '''
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype


# Load the Covertype dataset
covtype = fetch_covtype()
X, y = covtype.data, covtype.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_clf.predict(X_test)

import pickle
# Save your model as a pickle file
with open('dt_clf_model.pkl', 'wb') as f:
    pickle.dump(dt_clf, f)

''' 5. Evaluate your neural network and other models'''

from sklearn.metrics import classification_report
print('Decision Tree Report:')
print(classification_report(y_test, y_pred))

# %%
''' Random Forest Classifier: '''

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype


# Load the Covertype dataset
covtype = fetch_covtype()
X, y = covtype.data, covtype.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_clf.predict(X_test)

import pickle
# Save your model as a pickle file
with open('rf_clf_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

''' 5. Evaluate your neural network and other models'''
from sklearn.metrics import classification_report
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred))

# %%
''' Decision Tree Classifier is a simple and powerful classification algorithm that can be used as a baseline model.
    Random Forest Classifier can be used as an alternative baseline model to the decision tree classifier.
                                                                                                                    '''

# %%
''' 4. Use TensorFlow library to train a neural network that will classify the data '''
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("covtype.data")

# Split the dataset into features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Convert target variable to one-hot encoding
y = tf.keras.utils.to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the training and testing sets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the hyperparameters to be tuned
params = {
    'n_layers': [1, 2, 3],
    'n_neurons': [64, 128, 256],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30]
}

# Define the function to build and train the model
def build_model(n_layers, n_neurons, dropout_rate, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for layer in range(n_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

# Define the function to find the best set of hyperparameters
def find_best_params(params):
    best_score = 0
    best_params = None
    for n_layers in params['n_layers']:
        for n_neurons in params['n_neurons']:
            for dropout_rate in params['dropout_rate']:
                for learning_rate in params['learning_rate']:
                    for batch_size in params['batch_size']:
                        for epochs in params['epochs']:
                            model = build_model(n_layers, n_neurons, dropout_rate, learning_rate)
                            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
                            score = model.evaluate(X_test, y_test, verbose=0)[1]
                            if score > best_score:
                                best_score = score
                                best_params = {'n_layers': n_layers, 'n_neurons': n_neurons, 'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
    return best_params

# Find the best set of hyperparameters
best_params = find_best_params(params)

# Build and train the model with the best set of hyperparameters
best_model = build_model(best_params['n_layers'], best_params['n_neurons'], best_params['dropout_rate'], best_params['learning_rate'])
history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_test, y_test), verbose=0)

# Save the model as a pickle file
with open('NN_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

# Plot training & validation loss values
print('NN evaluation')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# %%
''' 5. Evaluate your neural network and other models'''

# Evaluate the performance of the model on the training set
train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
print('Training Accuracy:', train_acc)

# Evaluate the performance of the model on the testing set
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print('Testing Accuracy:', test_acc)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
