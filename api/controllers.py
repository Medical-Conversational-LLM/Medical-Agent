from flask import Response, request, abort, jsonify, current_app
from utils import to_stream_response, normalize_data
from .db import get_user_conversations, create_user as create_user_action, resolve_user_from_request, create_conversation, find_conversation, create_message, get_conversation_messages, update_message
from .run_graph_concurrently import run_graph_concurrently


def create_user():
    return jsonify(create_user_action())


def get_conversations(): 
    return get_user_conversations()


def post_conversation():
    user = resolve_user_from_request()

    json_data = request.form or request.get_json()
    json_data["user_id"] = user["id"]
    generator = to_stream_response(
        create_chat(json_data)
    )

    response = Response(generator, mimetype='text/event-stream')
 
    response.headers['Connection'] = 'keep-alive'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'

    return response


def create_chat(data):
    is_new_chat = "chatId" not in data or data["chatId"] is None

    if is_new_chat:
        conversation = create_conversation(
            title=data['message']['content'],
            user_id=data["user_id"]
        )

        user_message = create_message(
            data['message']['content'],
            conversation_id=conversation["id"],
            user_id=data["user_id"]
        )

        response_message = create_message(
            "",
            conversation_id=conversation["id"],
        )

        chat = {
            "chat": {
                **conversation,
                "user_message": user_message,
                "response_message": response_message,
            },
            "type": "NEW_CHAT"
        }

        yield chat
        chat_history = []  # Assuming empty chat history for a new chat
    else:
        conversation = find_conversation(int(data["chatId"]))

        if not conversation or conversation["user_id"] != data["user_id"]:
            abort("Unable to find conversation")

        user_message = create_message(
            data['message']['content'],
            conversation_id=conversation["id"],
            user_id=data["user_id"]
        )

        chat_history = get_conversation_messages(conversation["id"])
 
    settings = data.get("settings", {})
    temperature = settings.get("temperature", 1)
    top_k = settings.get("top_k", 1)
    top_p = settings.get("top_p", 1)
    max_length = settings.get("max_length", 256)

    query = data['message']['content']
    model = data['model'] if "model" in data else ""
    available_models = ['rag', 'self-reflective']
    if model not in available_models:
        model = "self-reflective"

    results = run_graph_concurrently(
        query=query,
        chat_history=chat_history,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        model=model
    )

    response_message = None
    for result in results:
        if isinstance(result, dict) and result.get("type") == "TOKEN":
            if not response_message:
                response_message = create_message(
                    "",
                    conversation_id=conversation["id"],
                )
            update_message(response_message["id"], result["message"])

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
