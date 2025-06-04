import requests
from bs4 import BeautifulSoup
import time

def scrape_and_extract_text(url, retries=2, delay=5):
    print(f"Scraping: {url}")
    for attempt in range(retries + 1):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; LangChainBot/0.1; +http://example.com/bot)'}
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')

            main_content_selectors = [
                'article[role="main"]',
                'main',
                'div.theme-doc-markdown',
                'div[class*="markdown"]',
                'div.content',
                'article'
            ]
            main = None
            for selector in main_content_selectors:
                main = soup.select_one(selector)
                if main:
                    break

            if main:
                for tag_selector in ['script', 'style', 'nav', 'footer', 'header', 'aside', '.toc', 'button[class*="theme-edit-this-page"]', 'div.docs-breadcrumbs', 'div.theme-doc-toc-mobile', 'div.theme-doc-sidebar-container', 'details.dropdown', 'summary.theme-edit-this-page', 'a.hash-link']: # More aggressive cleaning
                    for tag in main.select(tag_selector):
                        tag.decompose()
                parts = [p.get_text(separator=' ', strip=True) for p in main.find_all(['h1','h2','h3','h4','p','li','code','pre','td','th','span'])]
                text = "\n\n".join(filter(None, parts))

                if not text.strip() or len(text) < 200:
                    text = main.get_text(separator='\n', strip=True)
            else:
                print(f"  Warning: Could not find a specific main content element for {url}. Using body and cleaning common noise.")
                for tag_selector in ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript', 'button']:
                    for tag in soup.select(tag_selector):
                        tag.decompose()
                text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''

            print(f"  Successfully scraped. Text length: {len(text)}")
            return text.strip()

        except requests.exceptions.Timeout:
            print(f"  Timeout error scraping {url} on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            print(f"  Request error scraping {url} on attempt {attempt + 1}: {e}")
        except Exception as e:
            print(f"  Generic error processing {url} on attempt {attempt + 1}: {e}")

        if attempt < retries:
            print(f"  Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"  Failed to scrape {url} after {retries + 1} attempts.")
            return None
    return None