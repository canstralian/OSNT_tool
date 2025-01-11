import streamlit as st
import requests
from bs4 import BeautifulSoup

def fetch_page_title(url):
    """
    Fetches the title of the given URL.

    Args:
        url (str): The URL of the webpage.

    Returns:
        str: The title of the webpage or an error message.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else 'No title found'
            return title
        else:
            return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("OSINT Tool")
    st.write("Enter a URL to fetch its title:")

    url = st.text_input("URL")
    if url:
        title = fetch_page_title(url)
        st.write(f"Title: {title}")

if __name__ == "__main__":
    main()
