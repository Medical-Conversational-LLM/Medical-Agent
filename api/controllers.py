from flask import Response, request, abort, jsonify
from utils import to_stream_response, generate_id
from .db import get_user_conversations, create_user as create_user_action, resolve_user_from_request, create_conversation, find_conversation, create_message, get_conversation_messages, update_message
from .run_graph_concurrently import run_graph_concurrently


def create_user():
    return jsonify(create_user_action())


def get_conversations():
    return jsonify(get_user_conversations())


def post_conversation():
    user = resolve_user_from_request()

    json_data = request.form or request.get_json()
    json_data["user_id"] = user["id"]
    generator = to_stream_response(
        create_chat(json_data)
    )

    response = Response(generator, mimetype='text/event-stream')
    def on_close():
        print("cancelled request")

    response.call_on_close(on_close)
    response.headers['Connection'] = 'keep-alive'
    response.headers['Cache-Control'] = 'no-cache'

    return response


def create_chat(data):

    is_new_chat = "chatId" not in data
    if is_new_chat:
        conversation = create_conversation(
            title=data['message']['content'],
            user_id=data["user_id"]
        )
    else:
        conversation = find_conversation(int(data["chatId"]))

    print(conversation)

    user_message = create_message(
        data['message']['content'],
        conversation_id=conversation["id"],
        user_id=data["user_id"]
    )

    response_message = create_message(
        "",
        conversation_id=conversation["id"],
    )

    if is_new_chat:
        chat = {
            "chat": {
                **conversation,
                "user_message": user_message,
                "response_message": response_message,
            },
            "type": "NEW_CHAT"
        }

        yield chat

    if not conversation or conversation["user_id"] != data["user_id"]:
        abort("unable to find conversation")

    settings = data["settings"] if "settings" in data else dict()
    temperature = settings["temperature"] if "temperature" in settings else 1
    top_k = settings["top_k"] if "top_k" in settings else 1
    top_p = settings["top_p"] if "top_p" in settings else 1
    max_length = settings["max_length"] if "max_length" in settings else 256

    query = data['message']['content']

    results = run_graph_concurrently(
        query,
        temperature,
        top_k,
        top_p,
        max_length
    )

    for result in results:   
        if type(result) is dict and "type" in result and result["type"] == "TOKEN":
            update_message(
                response_message["id"],
                result["message"]
            )
        yield result


def get_conversation(id):
    user = resolve_user_from_request()
    conversation = find_conversation(id)
    if not conversation or conversation["user_id"] != user["id"]:
        abort("unable to find conversation")

    return {
        **conversation,
        "messages": get_conversation_messages(id)
    }
