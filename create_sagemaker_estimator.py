import boto3
import sagemaker
import sagemaker.tensorflow


# # Let's use Amazon S3
# s3 = boto3.resource('s3')
# # Print out bucket names
# for bucket in s3.buckets.all():
#     print(bucket.name)

sagemaker_session = sagemaker.local.LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

role = 'arn:aws:iam::036828958124:user/VladHondru' # sagemaker.get_execution_role()

# tf_estimator = sagemaker.tensorflow.estimator.TensorFlow(
#     # py_version="py37",
#     # entry_point="create_estimator.py",
#     # source_dir=".",
#     role=role,
#     # framework_version="2.4", #2.2
#     instance_count=1,
#     instance_type="local",
#     image_uri='deep-eye'
# )
tf_estimator = sagemaker.estimator.Estimator(
    # py_version="py37",
    # entry_point="create_estimator.py",
    # source_dir=".",
    role=role,
    sagemaker_session=sagemaker_session,
    # framework_version="2.4", #2.2
    output_path="s3://deep-eye-dev/dev_temp/outputs",
    instance_count=1,
    instance_type="local",
    image_uri='deep-eye:latest'
)

# tf_estimator.fit("s3://bucket/path/to/training/data")
# tf_estimator.fit()
tf_estimator.fit({
    'train': 's3://deep-eye-dev/dev_temp/val2',
    'common': 's3://deep-eye-dev/dev_temp/common',
    'eval': 's3://deep-eye-dev/dev_temp/val2'
})

tf_estimator.logs()

""" 
# sess = sagemaker.Session()

# from sagemaker.mxnet import MXNet

# # Configure an MXNet Estimator (no training happens yet)
# mxnet_estimator = MXNet('train.py',
#                         role='SageMakerRole',
#                         instance_type='local',
#                         instance_count=1,
#                         framework_version='1.2.1')

# # In Local Mode, fit will pull the MXNet container Docker image and run it locally
# mxnet_estimator.fit('s3://my_bucket/my_training_data/')

# # Alternatively, you can train using data in your local file system. This is only supported in Local mode.

# mxnet_estimator.fit('file:///tmp/my_training_data')
"""