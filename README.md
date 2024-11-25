# US_Health_Insurance

1) Problem statement
A problem statement for health insurance typically outlines challenges or gaps within the industry, providing a clear focus for analysis or solution development. Here's a general example:

Problem Statement: Health Insurance
The health insurance industry faces significant challenges in providing affordable, accessible, and comprehensive coverage to diverse populations. Despite advancements in medical technology and data analytics, the sector struggles with:

High Costs: Rising healthcare expenses make insurance premiums unaffordable for many individuals and families.
Accessibility Issues: A significant portion of the population, especially in rural or underprivileged areas, lacks access to quality health insurance plans.
Complexity and Transparency: Many customers find insurance plans and policies difficult to understand, leading to underutilization of benefits or dissatisfaction.
Fraud and Risk Management: Insurers face issues such as fraud detection, overclaiming, and accurately assessing risks, which impact profitability and service quality.
Personalization: There is a lack of tailored insurance products that cater to the specific health needs of individuals or demographic groups.
Technological Barriers: Many insurance companies lag in leveraging modern technologies like AI, blockchain, and big data for efficient claims processing, predictive analytics, and customer engagement.

## Live matarials docs

[link](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset?resource=download)



## How to run?

```bash
conda create -n insurance python=3.10.15 -y
```

```bash
conda activate insurance
     or 
source activate insurance
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```
## workflows
setup.py
data_access
mongodb
constant
components
entity
pipeline
ci/cd pipeline
Docker image
cloud_storage
demo.py
app.py


### Export the  environment variable
```bash


export MONGODB_URL="mongodb+srv://<username>:<password>...."

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```
# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 136566696263.dkr.ecr.us-east-1.amazonaws.com/mlproject

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO

    