# cc643861_pa2
CS 643-861 CLOUD COMPUTING 
PROGRAMMING ASSIGNMENT 2
Github Link: https://github.com/nithya2503/cc643861_pa2
Docker Hub Link: https://hub.docker.com/repository/docker/nithya252423/nk659wineapp/
STEPS TO FOLLOW
1.	To create a spark cluster in AWS
•	Login to your AWS account and navigate to EMR console
•	Create a key pair by navigating to EC2 – Network and security – key pairs, choose .pem format while downloading the key pair, mark the location of it to use it in future.
•	Go to the EMR console and create cluster, give a name to the cluster, choose emr-5.33.0
•	Choosing software configurations as Hadoop 2.10.1, Hive 2.3.7, Spark 2.4.7, Zeppelin 0.9.0 in the check box
•	Leave the core as default, m5.xlarge, for the task choose 3 tasks and 1 as core
•	Leave the rest of all the configurations as default.
•	Successful creation of cluster shows waiting as its status.
•	All the instances will be running in the EC2.
2.	Training the Machine Learning Model with 4 EC2 instances parallelly using Spark cluster.
o	Connect Master with SSH using CMD or powershell using ssh -I “key” user@Public IPV4 DNS
ssh -i "pa2.pem" ec2-user@ec2-34-228-231-249.compute-1.amazonaws.com
o	Change the user to root user using command ‘’’’’ sudo su ‘’’’’
o	To submit the job - spark-submit s3://winepredbuck/wine_qual_pred.py
o	The status of the job can be traced out in EMR UI application logs, the model will appear in the s3 bucket. arn:aws:s3:::winepredbuck
3.	Running Machine Learning Model using Docker
o	Install Docker in your machine according to the operating system of your machine
o	Create a Docker file with the set of instructions and build an image of it – docker build -t imagename (docker build -t nk659wineapp .)
o	Tag the image docker tag imagename username/imagename (docker tag nk659wineapp nithya252423/nk659wineapp)
o	Push the image to the docker hub: docker push username/imagename (docker push nithya252423/nk659wineapp)
o	Pull the image to your machine by docker pull username/imagename 
o	Trace the path of the test data, where the docker container mounts it.
o	Docker run -v /path to the testdata/: nk659wineapp testdata.csv
4.	Running the Machine learning model without using docker

o	Begin by cloning this repository to your local machine.
o	Ensure that you have a local Spark environment set up for running this application. If you don't have Spark set up yet, you can follow the instructions provided in the [official Spark documentation](https://spark.apache.org/docs/latest).
o	Navigate to the 'python file' folder within the cloned repository.
o	Place your test data in the ' C:\pa2-cc\data\csv' folder.
By following these steps, you'll be ready to run the trained machine learning model locally without relying on Docker.
