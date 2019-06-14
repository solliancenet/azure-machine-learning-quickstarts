
# Exercise 1: Setup New Project in Azure DevOps

## Task 1: Create New Project

1. Sign in to Azure DevOps
2. Select **Create project**

    ![Create new project in Azure DevOPs](images/01.png)

3. Provide Project Name: `mlops-quickstart` and select **Create**

    ![Provide Project Name](images/02.png)

## Task 2: Import Quickstart code from a Github Repo

1. Within the new project:

   a. Select **Repos** from left navigation bar
   
   b. Select **Import** from the content section
   
      ![Import Quickstart code from a Github Repo](images/03.png)
   
2. Provide the following Github URL: `https://github.com/solliancenet/mlops-starter.git` and select **Import**. This should import the code required for the quickstart.

    ![Provide the Github URL](images/04.png)

## Task 3: Update the build YAML file

1. Select and open the `azure-pipelines.yml` file
2. Select **Edit** and update the following variables: `resourcegroup`, `workspace` and `location`. If you are using your own Azure subscription, please provide names to use. If an environment is provided to you be sure to replace XXXXX in the values below with your unique identifier and update the `location` variable.

    ![Open build YAML file](images/05.png)

3. Select **Commit** to save your changes.

    ![Save your changes to YAML file](images/06.png)
  
## Task 4: Create new Service Connection

1. From the left navigation select **Project settings** and then select **Service connections**

  ![Open Service connections](images/07.png)

2. Select **New service connection** and then select **Azure Resource Manager**

  ![Open Azure Resource Manager](images/08.png)

3. Provide the following information in the `Add an Azure Resource Manager service connection` dialog box and then select **Ok**:
 
   a. Connection name: `quick-starts-sc`
   
   b. Subscription: Select the Azure subscription to use
   
   c. Resource Group: This value should match the value you provided in the `azure-pipelines.yml` file
   
    ![Add an Azure Resource Manager service connection](images/09.png)


# Exercise 2: Setup and Run the Build Pipeline

## Task 1: Setup Build Pipeline

1. From left navigation select **Pipelines, Builds** and then select **New pipeline**

    ![Setup Build Pipeline](images/10.png)
    
2. Select **Azure Repos Git** as your code repository

    ![Select your code repository](images/11.png)

3. Select **mlops-quickstart** as your repository

    ![Select mlops-quickstart as your repository](images/12.png)

4. Reivew the YAML file

    ![Reivew the YAML file](images/13.png)

## Task 2: Run the Build Pipeline

1. Select **Run** to start running your build pipeline

    ![Reivew the YAML file](images/14.png)

2. Monitor the build run. The build pipeline will take around *10-12 minutes* to run.

    ![Reivew the YAML file](images/15.png)

## Task 3: Review Build Artifacts




