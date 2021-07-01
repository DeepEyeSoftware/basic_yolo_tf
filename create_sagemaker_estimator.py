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

role = 'arn:aws:iam::036828958124:user/VladHondru'

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

tf_estimator.fit({
    'train': 's3://deep-eye-dev/dev_temp/val2',
    'common': 's3://deep-eye-dev/dev_temp/common',
    'eval': 's3://deep-eye-dev/dev_temp/val2'
})

tf_estimator.logs()
