from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import pickle


# Save Model In artifacts
def save_model(model, path):
    pickle.dump(model, path)


# Classification Class for custom training
class ClassificationModels:
    @staticmethod
    def logistic_regression_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = LogisticRegression(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def support_vector_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = SVC(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def decision_tree_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = DecisionTreeClassifier(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def k_neighbors_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = KNeighborsClassifier(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def random_forest_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = RandomForestClassifier(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def gradient_boosting_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = GradientBoostingClassifier(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def ada_boost_classifier(X_train, y_train, fit_model=False, **kwargs):
        model = AdaBoostClassifier(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model
