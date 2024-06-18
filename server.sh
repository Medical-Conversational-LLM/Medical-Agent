#!/bin/bash
 
export FLASK_APP=./web.py
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=8041
export FLASK_ENV=development

ENV="dev"

if [ $# -ge 1 ]; then
    ENV=$1
fi

echo "Running server in $ENV environment"


if [ "$ENV" == "prod" ]; then
    echo "Starting server in production mode..."
    gunicorn --workers 1 --timeout 500  --worker-class gevent --bind $FLASK_RUN_HOST:$FLASK_RUN_PORT $(basename $FLASK_APP .py):app
else
    echo "Starting server in development mode..."
    flask run
fi