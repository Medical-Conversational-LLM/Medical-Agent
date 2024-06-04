# import re

# # Original text
# original_text = """[Utility: 5

# The response appears to be somewhat relevant to the query, but it does not provide a direct analysis of the uncertainty and bias in DCE-MRI measurements using the spoiled gradient-recalled echo pulse sequence as requested. The response seems to be more of a general statement about the capabilities of small-animal imaging systems and pulse sequences, rather than a specific analysis of the uncertainty and bias in DCE-MRI measurements.]"""

# # Use regular expression to extract the desired text
# # cleaned_text = re.findall(r'Utility:\s\d', original_text)

# # print("[{}]".format(cleaned_text[0]))



from tinydb import TinyDB, Query
from utils import generate_id
from flask import request, abort

# user_table = TinyDB("./storage/cache/web-user.json")
conversations_table = TinyDB("./storage/cache/conversations.json")
# messages_table = TinyDB("./storage/cache/messages.json")


c = conversations_table.insert({})

print(
    conversations_table.get(doc_id=c)
)