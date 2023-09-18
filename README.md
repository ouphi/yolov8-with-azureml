# Train Yolov8 with AzureML

This repository provides an example showing how to train the yolov8 model with the az cli or the python SDK.

## az cli

You can find the detailed instructions to train the yolov8 model with the az cli [in this document](instructions-az-cli.md).

## Azure machine learning python SDK

[Here is a notebook](instructions-python-sdk.ipynb) showing how to train the yolov8 model with the python SDK.

## Deploy model for inference

### Register the model from the workspace UI
You can register the model resulting from a training job. 
Go to your job Overview and select "Register model".
Select model of type Unspecified type enable "Show all default outputs" > and select best.pt.
(Note that your training environment needs azureml-mlflow==1.52.0 and mlflow==2.4.2 to enable mlflow logging and being able to retrieve the model)

### Create the deployment
In `azureml/deployment.yaml`, specify your model

You can either specify a registered model.

```yaml
model: azureml:<your-model-name>:<version>
```

Or specify the relative path of a local .pt file:

```yaml
model:
  path: <model-relative-path-to-azureml-folder>
```

### Deploy your model for inference 

To deploy your endpoint in your azureml workspace:

Configure your default resource group and azureml workspace:

```bash
az configure --defaults group=$YOUR_RESOURCE_GROUP workspace=$YOUR_AZ_ML_WORKSPACE
```

```bash
./deploy-endpoint.sh
```

Note your endpoint name and score uri (you can retrieve them from the azure workspace).

### Test the endpoint and allocate traffic

To be able invoke our endpoint with an http client, you need to allocate traffic to your endpoint. (For more information [see this doc](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2&tabs=azure-cli#confirm-your-existing-deployment))

```bash
az ml online-endpoint show -n $ENDPOINT_NAME --query traffic
```

You can see that 0% is allocated to the blue deployment, so let's allocate 100% traffic to our unique blue deployment:

```bash
az ml online-endpoint update --name $ENDPOINT_NAME --traffic "blue=100"
```

Now you should be able to call your endpoint with curl.
You need to retrieve your endpoint key from the azure ml workspace in Endpoints > Consume > Basic consumption info. 

```bash
ENDPOINT_KEY=$YOUR_ENDPOINT_KEY
curl --request POST "$SCORING_URI" --header "Authorization: Bearer $ENDPOINT_KEY" --header 'Content-Type: application/json' --data '{"image_url": "https://ultralytics.com/images/bus.jpg"}'
```
