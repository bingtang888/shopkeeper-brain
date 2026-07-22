# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error


def main():

    # 1、实例化MinIO客户端
    client = Minio(
        endpoint="172.19.147.173:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    # 2、上传的文件地址
    source_file = "D:\\forAI\\project\\taylor.jpg"

    # 3、桶名
    bucket_name = "python-test-bucket"

    # 4、对象名字
    destination_file = "my-test-jpg.jpg"

    # 5、判断桶是否存在
    found = client.bucket_exists(bucket_name=bucket_name)
    if not found:
        # 5.1、创建桶
        client.make_bucket(bucket_name=bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # 6、上传文件
    client.fput_object(
        bucket_name=bucket_name,
        object_name=destination_file,
        file_path=source_file,
    )
    print("上传成功")

    # 7、获取访问地址
    url = client.presigned_get_object(
        bucket_name=bucket_name,
        object_name=destination_file,
    )
    print("访问地址:", url)


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
