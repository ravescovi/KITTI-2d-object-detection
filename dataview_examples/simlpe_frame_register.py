from allegroai import DatasetVersion, SingleFrame, FrameGroup


# See SingleFrame() for full set of arguments for the constructor
frame = SingleFrame(# this is where the frame is actually stored
                    source='https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg',
                    # This is how the browser will be able to fetch us this frame
                    # example: 's3://my_bucket/kaggle/3362/media/woof_meow.jpg'
                    # example: 's3://ip_of_minio:9000/bucket/kaggle/3362/media/woof_meow.jpg'
                    # example: 'gs://my_bucket/kaggle/3362/media/woof_meow.jpg'
                    # example: 'azure://my_bucket/kaggle/3362/media/woof_meow.jpg'
                    # the link below is a valid link, so we use it for this example
                    # preview_uri='https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg'
)

# Add annotation
# example bounding box at x=10,y=10 with width of 30px and height of 20px
# label of the bounding box is test
# See SingleFrame.add_annotation for full features and documentation. A few example below:
#   frame_class=None, poly2d_xy=None, poly3d_xyz=None, points2d_xy=None,
#   points3d_xyz=None, box2d_xywh=None, box3d_xyzwhxyzwh=None,
frame.add_annotation(box2d_xywh=(10, 10, 30, 20), labels=['test'])

# Add frame level metadata
frame.metadata = {'location': 'home',
                  'distance': '13.37',
                  'year': 2019}

# Create a dataset if it doesn't exists already
DatasetVersion.create_new_dataset('DATASET_TEST')

# Create a dataset version. Dataset version is a collection of frames and annotations,
# we could later freeze this version, and avoid accidental data overwriting or loss.
dataset = DatasetVersion.get_current(dataset_name='DATASET_TEST')

# Add or Update frames in the dataset
# after this function is executed, we will be able to see the new frame in the web-app
dataset.add_frames([frame, ])

print('We are done, see you next time')
