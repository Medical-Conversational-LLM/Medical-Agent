from flask import Flask
from flask_cors import CORS

from api.controllers import get_conversations, post_conversation, create_user, get_conversation
from logger import profiler

app = Flask(__name__)
CORS(app)

@app.route('/conversations', methods=['GET'])
def conversations():
    return get_conversations()


@app.route('/conversations', methods=['POST'])
def post_conversation_route():
    return post_conversation()


@app.route('/user', methods=['POST'])
def create_user_route():
    return create_user()


@app.route('/conversations/<int:conversation_id>', methods=['GET'])
def get_converastion_route(conversation_id):
    return get_conversation(conversation_id)


profiler.profile("TIME TAKEN TO START: ")

