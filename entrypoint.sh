#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
    
conda activate self-reflective

python /app/create_medical_records_index.py

python /app/download_nltk_resources.py

exec service nginx start

exec supervisord -c /etc/supervisor/conf.d/supervisord.conf -n