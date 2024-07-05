from tinydb import TinyDB, table
from cache.cache import Cache
from typing import Any
import hashlib


def hash(value: str):
    store_hash = hashlib.md5(value.encode())
    return store_hash.hexdigest()


class TinyDBCache(Cache):
    def put(self, store, id: Any):    
        db = TinyDB("./storage/cache/{}.json".format(hash(store)))
        db.insert(table.Document({}, doc_id=id))

    def has(self, store: str, id: str):
        try:        
            db = TinyDB("./storage/cache/{}.json".format(hash(store)))
            item = db.get(doc_id=id)
 
            return True if item is not None else False
        except:
            return False
