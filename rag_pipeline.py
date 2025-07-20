
import os,time
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import chunk_texts
from web_scraper import scrape_urls
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)


def get_all_links(start_url, max_links=50):
    visited = set()
    to_visit = [start_url]
    collected_links = []

    while to_visit and len(collected_links) < max_links:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            res = requests.get(url, timeout=5)
            visited.add(url)

            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            for link_tag in soup.find_all("a", href=True):
                href = urljoin(url, link_tag["href"])
                if href.startswith(start_url) and href not in visited:
                    to_visit.append(href)
                    collected_links.append(href)

                    if len(collected_links) >= max_links:
                        break
        except:
            continue

    return list(set(collected_links))

def build_index_from_urls(urls):
    data = scrape_urls(urls)

    chunks, metadata = [], []
    for url, content in data:
        for chunk in chunk_texts([content]):
            chunks.append(chunk)
            metadata.append({"source": url})

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")

    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump((chunks, metadata), f)


def retrieve_relevant_chunks(query, top_k=3):
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks, metadata = pickle.load(f)

    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [(chunks[i], metadata[i]) for i in I[0]]


def query_llm_groq(prompt, model="llama3-70b-8192", api_key="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"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    while True:
        res = requests.post(url, headers=headers, json=data)
        if res.status_code == 429:
            print("Rate limit hit. Waiting 5 seconds...")
            time.sleep(5)
            continue
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    # Your list of URLs to scrape and index
    url=get_all_links("https://kmit.in", max_links=200)
    # url2=get_all_links("https://bing.com/search?q=kmit", max_links=100)
    # url=url + url2
    url = list(set(url))  # Remove duplicates
    urls = [
        "https://www.kmit.in/",
        "https://www.kmit.in/aboutus/aboutus.php",
        "https://www.kmit.in/department/about_cse.php",
        "https://www.kmit.in/placements/placement.php",
        "https://www.kmit.in/examination/exam.php",
        "https://www.kmit.in/admissions/eapcet-lastrank.php",
        "https://www.kmit.in/admissions/admission-procedure.php",
        "https://collegedunia.com/college/13998-keshav-memorial-institute-of-technology-kmit-hyderabad",
        # "http://jntuhaac.in/Public/AutonomousCollegeInformation/",
        "https://www.collegedekho.com/colleges/kmit",
        "https://en.wikipedia.org/wiki/Keshav_Memorial_Institute_of_Technology",
        "https://www.shiksha.com/college/keshav-memorial-institute-of-technology-kmit-hyderabad-13998",
        "https://www.careers360.com/colleges/keshav-memorial-institute-of-technology-hyderabad",
        "https://www.indiatoday.in/education-today/colleges/story/keshav-memorial-institute-of-technology-kmit-hyderabad-admission-2023-2341232-2023-06-30",
    ]
    for u in url:
        urls.append(u)
    build_index_from_urls(set(urls))

# Add more embeddings to the existing FAISS index:
# new_embeddings = model.encode(new_chunks)
# index.add(new_embeddings)

# # Don’t forget to also update the `chunks.pkl` file
# chunks.extend(new_chunks)
# metadata.extend(new_metadata)

# # Save updated data
# faiss.write_index(index, "faiss_index/index.faiss")
# with open("faiss_index/chunks.pkl", "wb") as f:
#     pickle.dump((chunks, metadata), f)
# Example usage of the retrieval function:

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
# query = "What is artificial intelligence?"
# query_vec = model.encode([query])
# index = faiss.read_index("faiss_index/index.faiss")
# D, I = index.search(query_vec, 5)
# print(D, I)
# with open("faiss_index/chunks.pkl", "rb") as f:
#         chunks, metadata = pickle.load(f)
# for i in I[0]:
#     print("🔍 Retrieved Chunk:")
#     print(chunks[i])
#     print("🔗 Source:", metadata[i]['source'])
