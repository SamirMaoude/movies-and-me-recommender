#!/usr/bin/env bash
# Release the API on AWS Lambda through the staging and live aliases.
#
#   deploy IMAGE REVISION RUN_ID
#       Publish an ECR image, given by digest, as a new version. Check it on
#       staging, then on live; live returns to its previous version if its
#       check fails.
#   promote VERSION
#       Point live to a version published earlier, then check it.
#   status
#       Show the aliases and the latest versions.
#   url ALIAS
#       Print the public URL of live or staging.
set -euo pipefail

function_name=movies-and-me-recommender
python=${PYTHON:-python3}

url() {
  aws lambda get-function-url-config --function-name "$function_name" \
    --qualifier "$1" --query FunctionUrl --output text
}

check() {
  "$python" "$(dirname "$0")/check_api.py" --url "$(url "$1")" --timeout 120 \
    --favorite-id 27205 --expected-revision "$2" --expected-run-id "$3"
}

alias_version() {
  aws lambda get-alias --function-name "$function_name" --name "$1" \
    --query FunctionVersion --output text
}

point() {
  aws lambda update-alias --function-name "$function_name" --name "$1" \
    --function-version "$2" > /dev/null
  echo "$1 now serves version $2."
}

deploy() {
  local image=$1 revision=$2 run_id=$3 version previous
  aws lambda update-function-code --function-name "$function_name" \
    --image-uri "$image" > /dev/null
  aws lambda wait function-updated-v2 --function-name "$function_name"
  # The hash makes publishing fail if another update replaced this image.
  version=$(aws lambda publish-version --function-name "$function_name" \
    --code-sha256 "${image##*@sha256:}" \
    --description "commit $revision, model run $run_id" \
    --query Version --output text)
  aws lambda wait function-active-v2 --function-name "$function_name" \
    --qualifier "$version"
  point staging "$version"
  check staging "$revision" "$run_id"
  previous=$(alias_version live)
  echo "live served version $previous."
  point live "$version"
  if ! check live "$revision" "$run_id"; then
    point live "$previous"
    echo "The check failed on live, which is back on version $previous." >&2
    return 1
  fi
}

promote() {
  local version=$1 description
  local pattern='commit ([0-9a-f]{40}), model run ([0-9a-f]{32})'
  description=$(aws lambda get-function-configuration \
    --function-name "$function_name" --qualifier "$version" \
    --query Description --output text)
  if [[ ! $description =~ $pattern ]]; then
    echo "Version $version does not name its commit and model run." >&2
    return 1
  fi
  local revision=${BASH_REMATCH[1]} run_id=${BASH_REMATCH[2]}
  point live "$version"
  check live "$revision" "$run_id"
}

status() {
  local name
  echo "| Alias | Version | URL |"
  echo "|---|---|---|"
  for name in live staging; do
    echo "| $name | $(alias_version "$name") | $(url "$name") |"
  done
  echo
  echo "| Version | Description |"
  echo "|---|---|"
  aws lambda list-versions-by-function --function-name "$function_name" \
    --query "Versions[?Version!='\$LATEST'].[Version,Description]" --output text \
    | tail -n 5 | awk -F '\t' '{ print "| " $1 " | " $2 " |" }'
}

case "${1:-} $#" in
  "deploy 4") deploy "$2" "$3" "$4" ;;
  "promote 2") promote "$2" ;;
  "status 1") status ;;
  "url 2") url "$2" ;;
  *)
    echo "Usage: $0 deploy IMAGE REVISION RUN_ID | promote VERSION | status | url ALIAS" >&2
    exit 2
    ;;
esac
