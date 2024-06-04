#!/bin/bash


export FLASK_APP=./web.py
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=5001
export FLASK_ENV=development

flask run