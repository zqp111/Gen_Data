import data_utils 
from viz import Ax3DPose


data_dir = './data/h36m/dataset'

train_subject_ids = [1,6,7,8,9,11]
test_subject_ids = [5]
actions = ["walking", "eating", "smoking", "discussion",  "directions",
           "greeting", "phoning", "posing", "purchases", "sitting",
           "sittingdown", "takingphoto", "waiting", "walkingdog",
           "walkingtogether"]
one_hot = True

train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )

print("done reading data.")

print(train_set[(1, "walking", 1, "even")])

