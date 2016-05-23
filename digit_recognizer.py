import h2o
h2o.init()
#load data
train_set = h2o.upload_file("train.csv")
test_set = h2o.upload_file("test.csv")
#Define X and y
y = "label"
X = list(set(train_set.col_names) - set(["label"]))
train_set[y] = train_set[y].asfactor()
from h2o.estimators import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
#grid search and k-fold
hidden_opt = [[32,32],[32,16,8],[100]]
l1_opt = [1e-4,1e-3]
hyper_parameters = {"hidden":hidden_opt, "l1":l1_opt}
model_grid = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)
model_grid.train(x=X, y=y,
			distribution="multinomial", epochs=1000,
			training_frame=train_set, nfolds=5, stopping_rounds=3,
			stopping_tolerance=0.05,
			stopping_metric="misclassification")
#get the best model
gs = model_grid.sort_by("mse")
best = h2o.get_model("Grid_DeepLearning_py_2_model_python_1459310941902_2_model_4")
pred = best.predict(test_set)
label = pred["predict"]
#output
lb_df = label.as_data_frame(use_pandas=True)
lb_df.reset_index(inplace=True)
lb_df.columns = ["ImageId", "Label"]
lb_df["ImageId"] = lb_df["ImageId"] + 1
lb_df.to_csv("sub.csv", header=True, index=False)