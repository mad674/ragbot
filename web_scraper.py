import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return ""

def scrape_urls(url_list):
    all_texts = []
    for url in url_list:
        text = scrape_url(url)
        if text:
            all_texts.append((url, text))
    return all_texts
