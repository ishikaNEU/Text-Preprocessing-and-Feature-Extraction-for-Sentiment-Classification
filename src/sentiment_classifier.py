
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a sentiment classifier
def train_classifier(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the classifier
def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
