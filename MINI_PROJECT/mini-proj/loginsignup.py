import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import subprocess

# Function to create a SQLite database and table for users
def create_user_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(username, password):
    hashed_password = pbkdf2_sha256.hash(password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Function to authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    stored_password = c.fetchone()
    conn.close()
    if stored_password:
        return pbkdf2_sha256.verify(password, stored_password[0])
    return False

# Main function for login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Login successful!")
            # Redirect to another Streamlit app file
            subprocess.run(["streamlit", "run", "app.py"])
        else:
            st.error("Invalid username or password.")

# Main function for signup page
def signup():
    st.title("Signup")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Signup"):
        if new_password == confirm_password:
            insert_user(new_username, new_password)
            st.success("Signup successful! You can now login.")
        else:
            st.error("Passwords do not match.")

if __name__ == "__main__":
    create_user_table()
    page = st.sidebar.selectbox("Choose a page", ["Login", "Signup"])
    if page == "Login":
        login()
    else:
        signup()
