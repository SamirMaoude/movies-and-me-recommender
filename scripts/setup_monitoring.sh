#!/usr/bin/env bash
# Create or update the CloudWatch monitoring of the Lambda API; safe to re-run.
# Needs AWS administrator access, for example with AWS_PROFILE and AWS_REGION.
#
#   scripts/setup_monitoring.sh ALERT_EMAIL
#
# The email address receives a confirmation link from SNS on the first run.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 ALERT_EMAIL" >&2
  exit 2
fi
email=$1
function_name=movies-and-me-recommender
namespace=MoviesAndMe
dashboard="$(dirname "$0")/../monitoring/dashboard.json"
account=$(aws sts get-caller-identity --query Account --output text)

# Lambda reports the memory used in the REPORT line of each invocation.
aws logs put-metric-filter --log-group-name "/aws/lambda/$function_name" \
  --filter-name max-memory-used \
  --filter-pattern '[report="REPORT", request_label, request_id, duration_label, duration, duration_unit, billed_label, billed_label2, billed, billed_unit, size_label, size_label2, size, size_unit, used_label="Max", used_label2="Memory", used_label3="Used:", max_memory_used, ...]' \
  --metric-transformations "metricName=MaxMemoryUsed,metricNamespace=$namespace,metricValue=\$max_memory_used,unit=Megabytes"

topic=$(aws sns create-topic --name movies-and-me-alerts --query TopicArn --output text)
subscribed=$(aws sns list-subscriptions-by-topic --topic-arn "$topic" \
  --query "length(Subscriptions[?Endpoint=='$email'])" --output text)
if [ "$subscribed" = 0 ]; then
  aws sns subscribe --topic-arn "$topic" --protocol email \
    --notification-endpoint "$email" > /dev/null
  echo "Confirm the subscription from the email sent to $email."
fi

# Traffic is sparse: a period without requests must not raise an alarm.
alarm() {
  aws cloudwatch put-metric-alarm --alarm-name "$1" --alarm-description "$2" \
    --alarm-actions "$topic" --treat-missing-data notBreaching "${@:3}"
}
live=(--dimensions "Name=FunctionName,Value=$function_name"
  "Name=Resource,Value=$function_name:live")

alarm movies-and-me-live-5xx "The live API returned server errors." \
  --namespace AWS/Lambda --metric-name Url5xxCount "${live[@]}" \
  --statistic Sum --period 300 --evaluation-periods 1 \
  --comparison-operator GreaterThanOrEqualToThreshold --threshold 1
alarm movies-and-me-throttles "Requests were rejected: both instances were busy." \
  --namespace AWS/Lambda --metric-name Throttles \
  --dimensions "Name=FunctionName,Value=$function_name" \
  --statistic Sum --period 300 --evaluation-periods 1 \
  --comparison-operator GreaterThanOrEqualToThreshold --threshold 5
# A cold start takes about 3 seconds: only a slowdown lasting 15 minutes counts.
alarm movies-and-me-live-latency "The live API stayed slow for 15 minutes." \
  --namespace AWS/Lambda --metric-name UrlRequestLatency "${live[@]}" \
  --extended-statistic p95 --period 300 --evaluation-periods 3 \
  --comparison-operator GreaterThanThreshold --threshold 2000
alarm movies-and-me-live-traffic "More than 10 requests per second on live, which raises the cost." \
  --namespace AWS/Lambda --metric-name UrlRequestCount "${live[@]}" \
  --statistic Sum --period 300 --evaluation-periods 1 \
  --comparison-operator GreaterThanThreshold --threshold 3000
alarm movies-and-me-memory "An instance used more than 400 of its 512 MB." \
  --namespace "$namespace" --metric-name MaxMemoryUsed \
  --statistic Maximum --period 300 --evaluation-periods 1 \
  --comparison-operator GreaterThanThreshold --threshold 400
# The share is only meaningful once enough favorites have been received.
alarm movies-and-me-unknown-favorites "Over 15 minutes, more than half of the favorites sent to live were missing from the training catalog." \
  --evaluation-periods 1 --comparison-operator GreaterThanThreshold --threshold 0.5 \
  --metrics '[
    {"Id": "favorites", "ReturnData": false, "MetricStat": {"Period": 900, "Stat": "Sum",
      "Metric": {"Namespace": "MoviesAndMe", "MetricName": "Favorites",
        "Dimensions": [{"Name": "Alias", "Value": "live"}]}}},
    {"Id": "unknown", "ReturnData": false, "MetricStat": {"Period": 900, "Stat": "Sum",
      "Metric": {"Namespace": "MoviesAndMe", "MetricName": "UnknownFavorites",
        "Dimensions": [{"Name": "Alias", "Value": "live"}]}}},
    {"Id": "share", "ReturnData": true, "Label": "Unknown favorites share",
      "Expression": "IF(favorites >= 20, unknown / favorites, 0)"}
  ]'

aws cloudwatch put-dashboard --dashboard-name "$function_name" \
  --dashboard-body "$(sed "s/ACCOUNT_ID/$account/g" "$dashboard")" \
  --query 'DashboardValidationMessages' --output text
echo "Dashboard: https://eu-north-1.console.aws.amazon.com/cloudwatch/home?region=eu-north-1#dashboards/dashboard/$function_name"
