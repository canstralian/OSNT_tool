from your_module.components.chat_box import chat_box
import unittest
from unittest.mock import MagicMock
import streamlit as st

class TestChatBox(unittest.TestCase):

    def test_chat_box_renders_correctly(self):
        session_state_mock = MagicMock()
        config_mock = {"chat_key": "value"}

        # Mocking Streamlit's text input
        st.text_input = MagicMock()

        chat_box(session_state_mock, config_mock)

        # Assert that the text_input function was called
        st.text_input.assert_called_with("Enter your message:")

if __name__ == '__main__':
    unittest.main()
