from allegroai import DataView, Task, DatasetVersion


# get the frames
dataview = DataView()
dataview.add_query(dataset_name='KITTI 2D', version_name='training')
train_list, validation_list = dataview.split_to_lists(ratio=[80, 20])

# add labels:
for frame in train_list:
    frame.add_annotation(frame_class=['training'])

# add labels:
for frame in validation_list:
    frame.add_annotation(frame_class=['validation'])

dataset = DatasetVersion.create_version(
    dataset_name='KITTI 2D', version_name='training and validation', parent_version_names=['training'])
dataset.add_frames(train_list + validation_list)
