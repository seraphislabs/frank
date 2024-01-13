import os
import redis
import pymysql
import pickle

class ShorttermMemory:
    def __init__(self, host='localhost', port=6379):
        print("** ShorttermMemory: Initializing short-term memory")
        self.db = {}
        self.db[0] = redis.Redis(host=host, port=port, db=0)
        self.db[1] = redis.Redis(host=host, port=port, db=1)
        self.db[2] = redis.Redis(host=host, port=port, db=2)
        print("=> ShorttermMemory is initialized")

    def close(self):
        print("** ShorttermMemory: Closing short-term memory")
        self.db[0].close()
        self.db[1].close()
        self.db[2].close()
        print("=> ShorttermMemory is closed")

    def set_value(self, db, key, value, expiration=0):
        """ Set a value with an optional expiration time (in seconds). """
        if expiration > 0:
            self.db[db].set(key, value, ex=expiration)
        else:
            self.db[db].set(key, value)

    def get_value(self, db, key):
        """ Get a value and return it as a string. """
        value = self.db[db].get(key)
        if value is not None:
            return value.decode('utf-8')
        return None
    
    def get_face_embeddings(self):
        """ Get a value and return it as a string. """
        keyList = []
        valueList = []
        keys = self.db[1].keys('*')
        
        for key in keys:
            value = self.db[1].get(key)
            loaded = pickle.loads(key)
            keyList.append(loaded)
            print(str(loaded))
            valueList.append(value.decode('utf-8'))

        return keyList, valueList
    
    def get_all(self, db):
        """ Get all keys and values from a database. """
        keys = self.db[db].keys('*')
        return keys

    def exists(self, db, key):
        return self.db[db].exists(key)

    def delete_key(self, db, key):
        self.db[db].delete(key)

    def set_expiration(self, db, key, time):
        """ Set expiration time for a key (in seconds). """
        return self.db[db].expire(key, time)

    def get_expiration(self, db, key):
        """ Get the remaining time until key expiration (in seconds). """
        return self.db[db].ttl(key)

class LongtermMemeory:
    def __init__(self, host, user, password, db):
        print("** LongtermMemory: Initializing long-term memory")
        self.connection = pymysql.connect(host=host,
                                          user=user,
                                          password=password,
                                          db=db,
                                          cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        print("=> LongtermMemory is initialized")

    def execute_query(self, sql, params=None):
        """Execute a given SQL query with optional parameters and return the last inserted ID for INSERT operations or a list of rows for SELECT operations."""
        self.cursor.execute(sql, params or ())
        if sql.strip().upper().startswith("SELECT"):
            return self.cursor.fetchall()
        else:
            self.connection.commit()
            # If it's an INSERT operation, return the ID of the inserted row
            if sql.strip().upper().startswith("INSERT"):
                return self.cursor.lastrowid
            # For other non-SELECT operations, return an empty list or you can modify this to return row count or status
            return []

    def close(self):
        self.cursor.close()
        self.connection.close()

class MemoryInterface:
    shortterm = None
    longterm = None

    def __init__(self):
        print ("**** MemoryInterface: Initializing memory interface")
        self.shortterm = ShorttermMemory(host="localhost", port=6379)
        self.longterm = LongtermMemeory(host="localhost", user="frank", password="34Analsex1337!", db="frank")

        self.shortterm.set_value(0, "mode", "init")

        print ("=> MemoryInterface is initialized")
        pass

    def release(self):
        print("** MemoryInterface: Releasing memory interface")
        self.shortterm.close()
        self.longterm.close()
        print("=> MemoryInterface is released")
        pass