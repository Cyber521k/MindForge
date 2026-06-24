"""Web URL extraction and site crawling.

Tiered extraction approach (from design doc):
  1. Try Firecrawl (if API key detected -- highest quality)  [future]
  2. Fall back to Crawl4AI (local, handles JS)  [future]
  3. Fall back to Trafilatura (fast, no JS)  [future]
  4. Fall back to BeautifulSoup (manual, last resort)  [v1 default]

For v1, we use requests + BeautifulSoup as the primary method.
Firecrawl/Crawl4AI/Trafilatura can be added later by extending extract_url().
"""

import re
import hashlib
import logging
from urllib.parse import urljoin, urlparse, urldefrag

logger = logging.getLogger(__name__)


def extract_url(url, method="auto"):
    """Extract text content from a web URL.

    Uses a tiered approach:
      - method="auto": try BeautifulSoup (v1 default)
      - method="beautifulsoup": force BeautifulSoup
      - method="firecrawl": try Firecrawl (future, falls back to bs4)
      - method="trafilatura": try Trafilatura (future, falls back to bs4)

    Args:
        url: The URL to extract content from
        method: Extraction method ("auto", "beautifulsoup", "firecrawl", "trafilatura")

    Returns:
        dict with keys:
            - content: str, extracted text content
            - title: str, page title
            - method_used: str, the extraction method that succeeded
            - url: str, the source URL
    """
    import requests

    # Fetch the page
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return {
            "content": "",
            "title": "",
            "method_used": "none",
            "url": url,
            "error": str(e),
        }

    html = response.text

    # Try requested method first, then fall back to BeautifulSoup
    if method in ("firecrawl", "trafilatura"):
        # Future: try the specific method, fall back to bs4
        logger.info(f"Method '{method}' not yet implemented, falling back to BeautifulSoup")
        method = "beautifulsoup"

    if method in ("auto", "beautifulsoup"):
        result = _extract_with_beautifulsoup(html, url)
        result["url"] = url
        return result

    # Default fallback
    result = _extract_with_beautifulsoup(html, url)
    result["url"] = url
    return result


def _extract_with_beautifulsoup(html, url):
    """Extract text content from HTML using BeautifulSoup.

    Args:
        html: Raw HTML string
        url: Source URL

    Returns:
        dict with content, title, method_used keys
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove script, style, and other non-content tags
    for tag in soup(["script", "style", "noscript", "meta", "link", "header", "footer", "nav"]):
        tag.decompose()

    # Remove comments
    from bs4 import Comment
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Extract text with structure
    # Preserve heading structure for Q&A generation
    lines = []

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
        text = element.get_text(strip=True)
        if text:
            # Mark headings with a prefix for Q&A generation
            tag_name = element.name.lower()
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                lines.append(text)
                lines.append("")  # blank line after heading
            else:
                lines.append(text)

    content = "\n".join(lines)

    # Clean up excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()

    return {
        "content": content,
        "title": title,
        "method_used": "beautifulsoup",
    }


def crawl_site(base_url, max_pages=50, max_depth=3, url_pattern=None):
    """Crawl a website and extract content from multiple pages.

    Args:
        base_url: Starting URL for the crawl
        max_pages: Maximum number of pages to crawl (default 50)
        max_depth: Maximum crawl depth from the base URL (default 3)
        url_pattern: Optional regex pattern to filter URLs (e.g., r'docs.python.org/3/.*')

    Returns:
        list[dict]: Each dict has 'url', 'title', 'content', 'method_used', 'depth'
    """
    visited = set()
    to_visit = [(base_url, 0)]  # (url, depth)
    results = []
    compiled_pattern = re.compile(url_pattern) if url_pattern else None

    # Parse base domain for same-domain filtering
    base_domain = urlparse(base_url).netloc

    while to_visit and len(results) < max_pages:
        url, depth = to_visit.pop(0)

        # Skip if already visited
        if url in visited:
            continue

        # Skip if exceeds max depth
        if depth > max_depth:
            continue

        # De-fragment the URL
        url, _ = urldefrag(url)
        if url in visited:
            continue

        visited.add(url)

        # Check URL pattern filter
        if compiled_pattern and not compiled_pattern.search(url):
            continue

        # Check same domain
        if urlparse(url).netloc != base_domain:
            continue

        logger.info(f"Crawling [{len(results)+1}/{max_pages}] depth={depth}: {url}")

        # Extract content from this page
        page = extract_url(url, method="auto")
        page["depth"] = depth

        if page.get("content"):
            results.append(page)

            # Find links on this page for further crawling
            if depth < max_depth:
                links = _extract_links(page.get("_html", ""), url, base_domain)
                for link in links:
                    if link not in visited:
                        to_visit.append((link, depth + 1))

    return results


def _extract_links(html, base_url, base_domain):
    """Extract same-domain links from HTML.

    Args:
        html: HTML content (may be empty if not stored)
        base_url: The URL of the page
        base_domain: Domain to filter links by

    Returns:
        list[str]: Absolute URLs on the same domain
    """
    if not html:
        return []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Skip anchors, javascript, mailto
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Convert to absolute URL
        abs_url = urljoin(base_url, href)
        abs_url, _ = urldefrag(abs_url)

        # Filter by domain
        if urlparse(abs_url).netloc == base_domain:
            links.append(abs_url)

    return list(dict.fromkeys(links))  # deduplicate preserving order
