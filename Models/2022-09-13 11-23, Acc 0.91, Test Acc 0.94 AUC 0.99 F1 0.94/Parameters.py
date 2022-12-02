#isnotebook = True
#isgooglecolab = False
#shell = ZMQInteractiveShell

K_Length = 8
D_Length = 1200
H1 = 1
W1 = 1
conv_input_size = (1200,)
input_size = 1200
output_size = 1

criterion = 'gini'
class_weight = 'balanced'
min_impurity_decrease = 1e-06
max_leaf_nodes = None
max_features = 35
min_weight_fraction_leaf = 0.0001
min_samples_leaf = 1
min_samples_split = 2
max_depth = 50