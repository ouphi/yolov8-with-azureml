set -ex

export ENDPOINT_NAME=endpt-`echo $RANDOM`
az ml online-endpoint create --local -n $ENDPOINT_NAME -f azureml/endpoint.yaml

# <create_deployment>
az ml online-deployment create --local -n blue --endpoint $ENDPOINT_NAME -f azureml/deployment.yaml
# </create_deployment>

# <get_status>
az ml online-endpoint show -n $ENDPOINT_NAME --local
# </get_status>

# check if create was successful
endpoint_status=`az ml online-endpoint show --local --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --local --name blue --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_endpoint>
az ml online-endpoint invoke --local --name $ENDPOINT_NAME --request-file inference-sample-request.json
# </test_endpoint>

# <test_endpoint_using_curl>
SCORING_URI=$(az ml online-endpoint show --local -n $ENDPOINT_NAME -o tsv --query scoring_uri)

curl --request POST "$SCORING_URI" --header 'Content-Type: application/json' --data @inference-sample-request.json

# <get_logs>
#az ml online-deployment get-logs --local -n blue --endpoint $ENDPOINT_NAME
# </get_logs>

curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://ultralytics.com/images/bus.jpg"}' $SCORING_URI

# <delete_endpoint>
#az ml online-endpoint delete --local --name $ENDPOINT_NAME --yes
# </delete_endpoint>
