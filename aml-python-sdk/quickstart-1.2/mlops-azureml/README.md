
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

    ![Start your build pipeline](images/14.png)

2. Monitor the build run. The build pipeline will take around *10-12 minutes* to run.

    ![Monitor your build pipeline](images/15.png)

## Task 3: Review Build Artifacts

1. The build will publish an artifact named `devops-for-ai`. Select **Artifacts, devops-for-ai** to review the artifact contents.

    ![Select Artifacts, devops-for-ai to review the artifact contents](images/16.png)

2. Select **outputs, eval_info.json** and then select **Download**. The `eval_info.json` is the output from the *model evaluation* step and the information from the evaluation step will be later used in the release pipeline to deploy the model. Select **Close** to close the dialog.

    ![Download output from the model evaluation step](images/17.png)

3. Open the `eval_info.json` in a json viewer or a text editor and observe the information. The json output contains information such as if the model passed the evaluation step (`deploy_model`: *true or false*), and the name of the created image (`image_name`) to deploy.

    ![Review information the eval_info json file](images/18.png)

## Task 4: Review Build Outputs

1. Log in to [Azure Portal](https://portal.azure.com). Open your **Resource Group, Workspace, Models** section, and observe the registered model: `cost-estimator`.

    ![Review registered model in Azure Portal](images/19.png)

2. Open your **Resource Group, Workspace, Images** section and observe the deployment image created during the build pipeline: `cost-estimator-image`.

    ![Review deployment image in Azure Portal](images/18.png)
    
    
# Exercise 3: Setup the Release Pipeline

## Task 1: Create an Empty Job

1. Navigate to **Pipelines, Releases** and select **New pipeline**

    ![Create new Release Pipeline](images/19.png)

2. Select **Empty job**

    ![Select empty job](images/20.png)

3. Provide Stage name: `Develop & Test` and close the dialog.

    ![Provide stage name for the release stage](images/21.png)

## Task 2: Add Build Artifact

1. Select **Add an artifact**

    ![Add an artifact](images/22.png)

2. Select Source type: `Build`, Source (build pipeline): `mlops-quickstart`. *Observe the note that shows that the mlops-quickstart publishes the build artifact named devops-for-ai*. Finally, select **Add**

    ![Provide information to add the build artifact](images/23.png)
    
## Task 3: Add Tasks to "Deploy & Test" Stage

1. Open **View stage tasks** link

    ![Open view stage tasks link](images/24.png)

2. Open **Variables** tab

    ![Open variables tab](images/25.png)

3. Add three Pipeline variables as name - value pairs:

    a. Name: `aci_name` Value: `aci-cluster01`
    
    b. Name: `description` Value: `"Cost Estimator Web Service"` *note the double quotes around description value*
    
    c. Name: `service_name` Value: `cost-estimator-service`
    
      ![Add Pipeline variables](images/26.png)
        
4. Open **Tasks** tab

    ![Open view stage tasks link](images/27.png)
    
5. Select **Agent job** and select **Agent pool** to be `Hosted Ubuntu 1604`

    ![Change Agent pool to be Hosted Ubuntu 1604](images/28.png)

6. Select **Add a task to Agent job**, search for `Use Python Version`, and select **Add**

    ![Add Use Python Version task to Agent job](images/29.png)

7. Provide Display name: `Use Python 3.6` and Version spec: `3.6`

    ![Provide Display name and Version spec](images/30.png)

8. Select **Add a task to Agent job**, search for `bash`, and select **Add**
    
    ![Add Use Bash task to Agent job](images/31.png)

9. Provide Display name: `Install Requirements` and select **object browser ...** to provide Script Path.

    ![Provide Display name](images/32.png)

10. Navigate to **Linked artifacts/_mlops-quickstart/devops-for-ai/environment_setup** and select **install_requirements.sh**

    ![Provide Script Path](images/33.png)

11.





