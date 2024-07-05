#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
    
conda activate self-rag3

python /app/create_medical_records_index.py

python /app/download_nltk_resources.py

exec supervisord -c /etc/supervisor/conf.d/supervisord.conf -n

# docker run --gpus all  --entrypoint "" -it 19f002d88c69 /bin/bash