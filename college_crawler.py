"""
college_crawler.py
------------------
Web crawler for Government Polytechnic Aurangabad official website.
Routes queries to the correct sub-page and returns clean text for LLM context.
"""

import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.gpaurangabad.ac.in"

# Route map: keyword patterns → relative URL path
ROUTE_MAP = [
    (r"admission|seat|eligib|apply|enroll|registration|seats available", "/academics/admission/"),
    (r"placement|company|job|salary|recruit|hired|package|campus", "/training-and-placement/placement-list/"),
    (r"placement brochure", "/training-and-placement/placement-brochure/"),
    (r"syllabus|curriculum|subject|course content", "/academics/syllabus/"),
    (r"holiday|vacation|leave|break", "/academics/list-of-holidays/"),
    (r"notice|circular|announcement|news", "/category/notices/"),
    (r"contact|phone|address|email|call|locate|visit", "/about-us/contact/"),
    (r"history|about|established|founded|overview", "/about-us/history/"),
    (r"anti.?ragging|ragging", "/academics/anti-ragging/"),
    (r"academic.?regulation|exam.?rule|rule", "/academics/academic-regulation/"),
    (r"attendance|absent|leave policy", "/academics/attendance/"),
    (r"mechanical|ME ", "/department/mechanical-engineering/about-mechanical-engineering/"),
    (r"electrical|EE ", "/department/electrical-engineering/about-electrical-engineering/"),
    (r"civil|CE ", "/department/civil-engineering/about-civil-engineering/"),
    (r"electronics|ECE|EL ", "/department/electronics-engineering/about-electronics-engineering/"),
    (r"computer|CSE|IT ", "/department/computer-science-and-engineering/about-computer-science-and-engineering/"),
    (r"library|book|reading", "/facilities-and-services/central-library/"),
    (r"hostel|dorm|accomod|room", "/facilities-and-services/hostels/"),
    (r"gym|fitness|sport", "/facilities-and-services/gymnasium/"),
    (r"wifi|internet|network", "/facilities-and-services/wi-fi/"),
    (r"principal|dr\.|director|message", "/about-us/principals-message/"),
    (r"vision|mission|goal", "/about-us/vision-mission/"),
    (r"aicte|approval|recognition", "/approval/aicte/"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def _detect_route(query: str) -> str:
    """Return the best-matching sub-URL for a given query string."""
    query_lower = query.lower()
    for pattern, path in ROUTE_MAP:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return path
    return "/"  # Default: home page


def _fetch_page(url: str, timeout: int = 8) -> str:
    """
    Fetch a URL and return clean visible text stripped of nav/footer noise.
    Returns an empty string if the page is unreachable.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Crawler] Failed to fetch {url}: {exc}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace / blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def fetch_context_for_query(query: str) -> dict:
    """
    Main entry point.
    Returns a dict with:
      - 'url': the page that was crawled
      - 'content': cleaned text (up to 3000 chars to stay within LLM context)
      - 'status': 'ok' | 'down' | 'empty'
    """
    path = _detect_route(query)
    full_url = BASE_URL + path
    print(f"[Crawler] Routing query to → {full_url}")

    content = _fetch_page(full_url)

    if not content:
        # Try the homepage as a last resort
        content = _fetch_page(BASE_URL)
        if not content:
            return {"url": full_url, "content": "", "status": "down"}

    # Trim to avoid blowing up the LLM context window
    trimmed = content[:3000]
    return {
        "url": full_url,
        "content": trimmed,
        "status": "ok" if trimmed else "empty",
    }
