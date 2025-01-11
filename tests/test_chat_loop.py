from your_module.components.chat_loop import chat_loop
import unittest
from unittest.mock import MagicMock
import streamlit as st

class TestChatLoop(unittest.TestCase):

    def test_chat_loop_renders_correctly(self):
        session_state_mock = MagicMock()
        config_mock = {"chat_key": "value"}

        # Mocking Streamlit's write function
        st.write = MagicMock()

        chat_loop(session_state_mock, config_mock)

        # Assert that the chat loop writes to the screen
        st.write.assert_called_with("Chat Loop Running...")

if __name__ == '__main__':
    unittest.main()
