# Project Workflow:

## Step 1: Real-time Data Ingestion with Apache Kafka Streams

1. Launch and stream real-time data from the 'customer_churn.csv' file using Apache Kafka Streams.

## Step 2: Data Preprocessing with Machine Learning Libraries

2. Perform necessary data preprocessing using libraries such as Sklearn, PySpark MLib, or PyTorch.

## Step 3: Supervised Machine Learning Training

3. Train supervised machine learning models (at least 3 models) on the 'customer_churn.csv' training dataset.

## Step 4: Model Serialization and Storage

4. Save the best-performing model in .pkl format.

## Step 5: Real-time Prediction using the Trained Model

5. Utilize the prepared, trained, and saved model to predict in real-time whether a customer will leave the institution or not based on the 'new_customers.csv' test data.

## Step 6: Results Presentation with Web Application Dashboard

6. Present the results in the form of a web application dashboard.

## Step 7: Project Upload to GitHub

7. Upload the entire project to GitHub for collaboration and version control.

# Tools and Technologies:

- **Libraries:** Apache Kafka Streams, PySpark MLib, Sklearn, PyTorch, Pandas, Matplotlib
- **Frameworks:** Flask, Django
- **Languages:** Python, Java, JavaScript
- **Editors:** IntelliJ IDEA, Eclipse, VsCode
- **Operating Systems:** Unix, MacOS, or Windows

# Data Description:

- **Name:** Name of the latest contact at Company
- **Age:** Customer Age
- **Total_Purchase:** Total Ads Purchased
- **Account_Manager:** Binary 0=No manager, 1= Account manager assigned
- **Years:** Total Years as a customer
- **Num_sites:** Number of websites that use the service.
- **Onboard_date:** Date that the name of the latest contact was onboarded
- **Location:** Client HQ Address
- **Company:** Name of Client Company
- **Churn:** Target (label)

# Data Source:

[Customer Churn Spark Notebook](https://github.com/Shantanu-Gupta-au16/Spark-Mini-Projects/blob/master/Customer%20Churn%20using%20Spark.ipynb)

