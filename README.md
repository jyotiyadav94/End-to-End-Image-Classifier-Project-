End-to-End-MLOPS-Image-Classifier-Project
==============================

Computer Vision Projects

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# AWS-CICD-Deployment-with-Github-Actions

1. Login to AWS console.
2. Create IAM user for deployment

## with specific access

1. **EC2 access**: It is a virtual machine.
2. **ECR**: Elastic Container Registry to save your Docker image in AWS.

## Description: About the deployment

1. Build Docker image of the source code.
2. Push your Docker image to ECR.
3. Launch your EC2.
4. Pull your image from ECR in EC2.
5. Launch your Docker image in EC2.

## Policy

1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess

3. Create ECR repo to store/save Docker image
   - Save the URI: `637423621763.dkr.ecr.us-east-1.amazonaws.com/mlproject`
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install Docker in EC2 Machine:

### Optional

```bash
sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

6. Configure EC2 as self-hosted runner:
- Settings > Actions > Runner > New self-hosted runner > Choose OS > Then run commands one by one

7. Setup GitHub secrets:

- AWS_ACCESS_KEY_ID=
- AWS_SECRET_ACCESS_KEY=
- AWS_REGION=us-east-1
- AWS_ECR_LOGIN_URI=demo >> 566373416292.dkr.ecr.ap-south-1.amazonaws.com
- ECR_REPOSITORY_NAME=simple-app

8. Configure Git:
git config --global user.name "jyotiyadav"


