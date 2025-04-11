# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
from typing import List
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Initialization
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are a helpful assistant for internet users called WebsiteDetailsAI. "
system_message += "You analyzes the contents of several relevant pages from a company website \
and creates detailed answers about the company for prospective customers, investors and recruits. "
system_message += "Give detailed summary answers and social media links for any website and \
answer questions based on the information gathered. "
system_message += "You are able to call a function to get the details of a website. "
system_message += "You are able to call a function to get the social media links of a website. "
system_message += "Always be accurate. Respond in markdown. If you don't know the answer, say so."

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(arguments)

        if tool_name == "get_website_details":
            print("Tool get_website_details called with: ", arguments.get('destination_website_url'))
            url = arguments.get('destination_website_url')
            if not url:
                return {"error": "No URL provided for get_website_details"}
            tool_response = get_website_details(url)
        elif tool_name == "get_social_media_links":
            print("Tool get_social_media_links called with: ", arguments.get('url'))
            url = arguments.get('url')
            if not url:
                return {"error": "No URL provided for get_social_media_links"}
            tool_response = get_social_media_links(url)
        else:
            tool_response = {"error": "Unknown tool called"}

        response_message = {
            "role": "tool",
            "content": json.dumps(tool_response),
            "tool_call_id": tool_call.id
        }
        messages.append(message)
        messages.append(response_message)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

def get_website_details(destination_website_url):
    destination_website_url = destination_website_url.lower()
    return get_all_details(destination_website_url)

website_details_function = {
    "name": "get_website_details",
    "description": "Get the all details of a destination website url. Call this whenever you need to know more about a website, \
          for example when a customer asks 'What does this website do' or 'Tell me more about this website'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_website_url": {
                "type": "string",
                "description": "The website the customer wants to have more informatin and details about",
            },
        },
        "required": ["destination_website_url"],
        "additionalProperties": False
    }
}

social_media_links_function = {
    "name": "get_social_media_links",
    "description": "Extracts social media links from a given webpage URL. Call this to retrieve a list of social media links available on the website.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to analyze for social media links."
            }
        },
        "required": ["url"],
        "additionalProperties": False
    }
}

# And this is included in a list of tools:
tools = [
    {"type": "function", "function": website_details_function},
    {"type": "function", "function": social_media_links_function}
]

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include gather information and details about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for information and details about the company, \
respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

def get_all_details(url):
    print("Getting all details for URL: ", url)
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

def take_screenshot(url, output_path):
    """
    Takes a screenshot of the given webpage and saves it to the specified output path.

    Args:
        url (str): The URL of the webpage to capture.
        output_path (str): The file path to save the screenshot.
    """
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--start-maximized')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        driver.save_screenshot(output_path)
        print(f"Screenshot saved to {output_path}")
    finally:
        driver.quit()

def get_social_media_links(url):
    """
    Extracts social media links from the given webpage.

    Args:
        url (str): The URL of the webpage to analyze.

    Returns:
        list: A JSON list containing social media site names and their URLs.
    """
    website = Website(url)
    social_media_sites = ["facebook.com", "twitter.com", "linkedin.com", "instagram.com", "youtube.com"]
    social_links = []

    for link in website.links:
        for site in social_media_sites:
            if site in link:
                social_links.append({"site": site.split('.')[0].capitalize(), "url": link})

    return social_links

# Example usage:
# social_links = get_social_media_links("https://example.com")
# print(json.dumps(social_links, indent=2))

gr.ChatInterface(fn=chat, type="messages").launch(inbrowser=True, share=True, debug=True)