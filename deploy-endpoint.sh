set -ex

export ENDPOINT_NAME=endpt-`echo $RANDOM`
az ml online-endpoint create -n $ENDPOINT_NAME -f azureml/endpoint.yaml
az ml online-deployment create -n blue --endpoint $ENDPOINT_NAME -f azureml/deployment.yaml
az ml online-endpoint show -n $ENDPOINT_NAME
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri)
echo "Endpoint name: $ENDPOINT_NAME"
echo "Scoring uri: $SCORING_URI"
