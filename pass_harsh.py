#import streamlit_authenticator as stauth

from streamlit_authenticator.utilities.hasher import Hasher

# Usernames and passwords
usernames = ['user1', 'user2']
passwords = ['password1', 'password2']

# Hash passwords
#hashed_passwords = stauth.Hasher(passwords).generate()
hashed_passwords = Hasher(passwords).generate()

# Print hashed passwords
for username, password in zip(usernames, hashed_passwords):
    print(f"{username}: {password}")
