import os
import boto3

# ── Beginner Configuration ───────────────────────────────────────────────────
APP_NAME = "synapsemed"
REGION = "us-east-1"
S3_BUCKET = f"{APP_NAME}-storage-bucket-beginner"
AMI_ID = "ami-0c7217cdde317cfec" 

print(f"[*] Preparing deployment for {APP_NAME} in region {REGION}...")

# ── Step 1: Create an S3 Bucket (infinite folder storage) ─────────────────────
s3 = boto3.client("s3", region_name=REGION)
try:
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=S3_BUCKET)
    else:
        s3.create_bucket(
            Bucket=S3_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
    print(f"[SUCCESS] S3 Bucket created: s3://{S3_BUCKET}")
except Exception as e:
    print(f"[INFO] S3 Bucket already exists or skipped: {e}")

# ── Step 2: Launch an EC2 Instance (rented cloud computer) ────────────────────
ec2 = boto3.resource("ec2", region_name=REGION)
try:
    print("[*] Requesting a new EC2 server from AWS...")
    instances = ec2.create_instances(
        ImageId=AMI_ID,
        MinCount=1,
        MaxCount=1,
        InstanceType="t2.micro",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"{APP_NAME}-server"}]
            }
        ]
    )
    
    instance = instances[0]
    print("[*] Waiting for the server to boot up...")
    instance.wait_until_running()
    instance.reload()
    
    print(f"[SUCCESS] EC2 Instance created: {instance.id}")
    print(f"[SUCCESS] Server Public IP: {instance.public_ip_address}")
    print(f"[SUCCESS] Server Public DNS: {instance.public_dns_name}")
except Exception as e:
    print(f"[ERROR] Failed to start EC2 server: {e}")

# ── Step 3: Beginner guide on Docker & ECR ────────────────────────────────────
print("\n" + "="*50)
print("LEARNING STEP: DOCKER & ECR COMMANDS")
print("="*50)
print("To upload your project code to the cloud, run these 4 CLI commands:")
print(f"1. Create ECR library:\n   aws ecr create-repository --repository-name {APP_NAME}")
print(f"\n2. Pack your code inside a Docker box:\n   docker build -t {APP_NAME} -f ../backend/Dockerfile ../backend")
print(f"\n3. Tag your Docker box (replace <ACCOUNT_ID> with your AWS ID):\n   docker tag {APP_NAME}:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/{APP_NAME}:latest")
print(f"\n4. Push the Docker box to AWS ECR:\n   docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/{APP_NAME}:latest")
print("="*50 + "\n")
