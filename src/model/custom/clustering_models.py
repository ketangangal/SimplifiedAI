from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pickle


# Save Model In artifacts
def save_model(model, path):
    pickle.dump(model, path)


# Clustering Class for custom training
class ClusteringModels:
    @staticmethod
    def kmeans_clustering(X, fit_model=False, **kwargs):
        model = KMeans(**kwargs)
        if fit_model:
            y_pred = model.fit_predict(X)
            return model, y_pred
        else:
            return model

    @staticmethod
    def dbscan_clustering(X, fit_model=False, **kwargs):
        model = DBSCAN(**kwargs)
        if fit_model:
            y_pred = model.fit_predict(X)
            return model, y_pred
        else:
            return model

    @staticmethod
    def agglomerative_clustering(X, fit_model=False, **kwargs):
        model = AgglomerativeClustering(**kwargs)
        if fit_model:
            y_pred = model.fit_predict(X)
            return model, y_pred
        else:
            return model
