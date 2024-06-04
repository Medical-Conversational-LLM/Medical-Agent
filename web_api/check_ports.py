import socket
import os

def find_available_port(ports):
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:  # Port is available
                return port
    return None


ports_to_try = [5001, 5001, 5002, 5003, 5004]
port = find_available_port(ports_to_try)
 
if port is not None:
    print(f"Running on port {port}")
    os.environ['FLASK_RUN_PORT'] = str(port)