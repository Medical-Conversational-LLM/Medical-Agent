from tinydb import TinyDB, Query
from utils import generate_id
from flask import request, abort
import datetime

user_table = TinyDB("./storage/cache/web-user.json")
conversations_table = TinyDB("./storage/cache/conversations.json")
messages_table = TinyDB("./storage/cache/messages.json")


def get_conversation_messages(id):

    Message = Query()

    messages = messages_table.search(
        Message.conversation_id == id)

    return [
        {**message, "id": message.doc_id} for message in messages
    ]


def get_user_conversations():
    user = resolve_user_from_request()

    Conversation = Query()

    conversations = conversations_table.search(
        Conversation.user_id == user["id"])
    conversations.reverse()

    return [conversation_from_record(conversation) for conversation in conversations]


def resolve_user_from_request():
    auth_header = request.headers.get('Authorization')

    if not auth_header:
        abort(404, description="Authorization failed")

    access_token = auth_header.split(
        " ")[1] if " " in auth_header else auth_header
    User = Query()

    user = user_table.search(User.access_token == access_token)

    if not user or len(user) == 0:
        abort(404, description="Authorization failed")

    return {
        **user[0],
        "id": user[0].doc_id
    }


def create_user():
    access_token = generate_id(32)
    user = {
        "name": "User",
        "access_token": access_token
    }

    user_table.insert(user)

    return user


def conversation_from_record(record):
    if not record:
        return None

    return {
        **record,
        "id": record.doc_id
    }


def create_conversation(title, user_id):
    id = conversations_table.insert({
        "title": title,
        "user_id": user_id
    })

    return find_conversation(id)


def find_conversation(id):
    if type(id) is not int:
        return None

    return conversation_from_record(
        conversations_table.get(doc_id=id)
    )


def create_message(content: str, conversation_id, user_id=None):
    id = messages_table.insert({
        "content": content,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()


    })

    message = messages_table.get(doc_id=int(id))

    return {
        **message,
        "id": message.doc_id
    }


def update_message(id, content):
    messages_table.update({
        "content": content,
    }, doc_ids=[id])

    message = messages_table.get(doc_id=int(id))

    return {
        **message,
        "id": message.doc_id
    }
