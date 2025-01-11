from unittest.mock import patch
import unittest
from your_module import fetch_page_title

class TestURLFunctions(unittest.TestCase):

    @patch('requests.get')
    def test_fetch_page_title(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><head><title>Test Title</title></head><body></body></html>"
        title = fetch_page_title("https://example.com")
        self.assertEqual(title, "Test Title")

    @patch('requests.get')
    def test_fetch_page_title_invalid_url(self, mock_get):
        mock_get.return_value.status_code = 404
        title = fetch_page_title("https://invalidurl.com")
        self.assertEqual(title, "Error: Received status code 404")

if __name__ == '__main__':
    unittest.main()
