# Using This File for dropdown menu
from enum import Enum

TWO_D_GRAPH_TYPES = ["Selet Any", "Bar Graph", "Histogram", "Scatter Plot", "Pie Chart", "Line Chart", "Box Plot",
                     "Dist Plot", "Heat Map"]
TWO_D_GRAPH_TYPES_2 = ["Selet Any", "Scatter Plot", "Line Chart", "Heat Map"]
THREE_D_GRAPH_TYPES = ["3D Axes", "3D Scatter Plot", "3D Surface Plot", "3D Bubble Charts"]

FEATURE_SELECTION_METHODS_CLASSIFICATION = ["Find Constant Features", "Mutual Info Classification",
                                            "Extra Trees Classifier", "Correlation", "Forward Selection",
                                            "Backward Elimination"]

FEATURE_SELECTION_METHODS_RGRESSOR = ["Find Constant Features", "Mutual Info Regressor",
                                      "Extra Trees Regressor", "Correlation", "Forward Selection",
                                      "Backward Elimination"]

FEATURE_SELECTION_METHODS_CURSOR = ["Find Constant Features", "Correlation"]

NUMERIC_MISSING_HANDLER = ['Mean', 'Median', 'Arbitrary Value', 'Interpolate']
OBJECT_MISSING_HANDLER = ['Mode', 'New Category', 'Select Exist']

SUPPORTED_DATA_TYPES = ['object', 'int64', 'float64', 'bool', 'datetime64', 'category', 'timedelta']
ENCODING_TYPES = ['Label/Ordinal Encoder', 'Hash Encoder', 'Binary Encoder', 'Base N Encoder', 'One Hot Encoder']
SUPPORTED_SCALING_TYPES = ['MinMax Scaler', 'Standard Scaler', 'Max Abs Scaler', 'Robust Scaler',
                           'Power Transformer Scaler']

PROJECT_TYPES = [
    {"id": 1, "name": "Regression"},
    {"id": 2, "name": "Classification"},
    {"id": 3, "name": "Clustering"}
]


class ProjectActions(Enum):
    DELETE_COLUMN = 1
    CHANGE_DATA_TYPE = 2
    COLUMN_NAME_CHANGE = 3
    ENCODING = 4
    SCALING = 5
    PCA = 6
    CUSTOM_SCRIPT = 7


REGRESSION_MODELS = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor',
                     'RandomForestRegressor', 'SVR', 'AdaBoostRegressor', 'GradientBoostingRegressor']
CLASSIFICATION_MODELS = ['LogisticRegression', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier',
                         'RandomForestClassifier', 'GradientBoostClassifier', 'AdaBoostClassifier']
CLUSTERING_MODELS = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']

ALL_MODELS = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor',
              'RandomForestRegressor', 'SVR', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'LogisticRegression',
              'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier',
              'RandomForestClassifier', 'GradientBoostClassifier', 'AdaBoostClassifier', 'KMeans', 'DBSCAN',
              'AgglomerativeClustering']

TIMEZONE = [{"-12": "(GMT -12:00) Eniwetok, Kwajalein"}, {"-11": "(GMT -11:00) Midway Island, Samoa"},
            {"-10": "GMT -10:00) Hawaii"}, {"-9": "(GMT -9:00) Alaska"},
            {"-8": "(GMT -8:00) Pacific Time (US &amp; Canada)"}, {"-7": "(GMT -7:00) Mountain Time (US &amp; Canada)"},
            {"-6": "(GMT -6:00) Central Time (US &amp; Canada), Mexico City"},
            {"-5": "(GMT -5:00) Eastern Time (US &amp; Canada), Bogota, Lima"}, {"-4.5": "(GMT -4:30) Caracas"},
            {"-4": "(GMT -4:00) Atlantic Time (Canada), La Paz, Santiago"}, {"-3.5": "(GMT -3:30) Newfoundland"},
            {"-3": "(GMT -3:00) Brazil, Buenos Aires, Georgetown"}, {"-2": "(GMT -2:00) Mid-Atlantic"},
            {"-1": "(GMT -1:00 hour) Azores, Cape Verde Islands"},
            {"0": "(GMT) Western Europe Time, London, Lisbon, Casablanca, Greenwich"},
            {"1": "(GMT +1:00 hour) Brussels, Copenhagen, Madrid, Paris"},
            {"2": "(GMT +2:00) Kaliningrad, South Africa, Cairo"},
            {"3": "(GMT +3:00) Baghdad, Riyadh, Moscow, St. Petersburg"}, {"3.5": "(GMT +3:30) Tehran"},
            {"4": "(GMT +4:00) Abu Dhabi, Muscat, Yerevan, Baku, Tbilisi"}, {"4.5": "(GMT +4:30) Kabul"},
            {"5": "(GMT +5:00) Ekaterinburg, Islamabad, Karachi, Tashkent"},
            {"5.5": "(GMT +5:30) Mumbai, Kolkata, Chennai, New Delhi"}, {"5.75": "(GMT +5:45) Kathmandu"},
            {"6": "(GMT +6:00) Almaty, Dhaka, Colombo"}, {"6.5": "(GMT +6:30) Yangon, Cocos Islands"},
            {"7": "(GMT +7:00) Bangkok, Hanoi, Jakarta"}, {"8": "(GMT +8:00) Beijing, Perth, Singapore, Hong Kong"},
            {"9": "(GMT +9:00) Tokyo, Seoul, Osaka, Sapporo, Yakutsk"}, {"9.5": "(GMT +9:30) Adelaide, Darwin"},
            {"10": "(GMT +10:00) Eastern Australia, Guam, Vladivostok"},
            {"11": "(GMT +11:00) Magadan, Solomon Islands, New Caledonia"},
            {"12": "(GMT +12:00) Auckland, Wellington, Fiji, Kamchatka"}]

OPTIMIZERS = ['Adam', 'AdaGrad', 'AdaMax', 'RMSProps']
ACTIVATION_FUNCTIONS = ['ReLU', 'ELU', 'LeakyReLU', 'Softmax', 'PReLU', 'SELU', 'Tanh', 'Softplus', 'Softmin',
                        'Sigmoid', 'RReLU']
REGRESSION_LOSS = ['MAE', 'MSE', 'Huber Loss', 'Smoth L1', 'BCEWithLogitsLoss', 'CrossEntropyLoss']
POOLING = ['MaxPool2d', 'AvgPool2d']