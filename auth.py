import bcrypt
from db import users_collection
from session import set_session

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed_password, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup_user(email, password,username):
    if users_collection.find_one({"email": email}):
        return False
    hashed_password = hash_password(password)
    users_collection.insert_one({"email": email, "password": hashed_password,'username':username})
    return True

def login_user(email, password):
    user = users_collection.find_one({"email": email})
    print(user)
    if user and check_password(user['password'], password):
        set_session(user['username'])
        return True
    return False,{}
