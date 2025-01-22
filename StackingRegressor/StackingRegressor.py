from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


best_SVR = SVR(kernel='rbf', epsilon=1, C=100, gamma=0.01)

best_GradientBoostingRegressor = GradientBoostingRegressor(learning_rate=0.15, max_depth=3, n_estimators=200,
                                                           min_samples_split=5, max_features=5, random_state=42,
                                                           min_samples_leaf=1, subsample=0.5)

best_MLPRegressor = MLPRegressor(hidden_layer_sizes=36, activation='relu', solver='adam', alpha=0.01,
                                 learning_rate='invscaling', learning_rate_init=0.0001, max_iter=800000,
                                 shuffle=True, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

best_RandomForestRegressor = RandomForestRegressor(max_depth=7, max_features=7, min_samples_leaf=2, min_samples_split=5,
                                                   n_estimators=100, random_state=42, max_leaf_nodes=30,
                                                   min_impurity_decrease=0.04, bootstrap=False)

best_XGBRegressor = XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=7, n_estimators=200, gamma=0)

estimators = [('SVR', best_SVR),
              ('GBR', best_GradientBoostingRegressor),
              ('MLP', best_MLPRegressor),
              ('RF', best_RandomForestRegressor),
              ('XGB', best_XGBRegressor)]

stacking_regressor = StackingRegressor(estimators=estimators)
