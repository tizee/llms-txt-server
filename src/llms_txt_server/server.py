import os
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urlparse
from abc import ABC, abstractmethod

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field, HttpUrl

from mcp.server.fastmcp import FastMCP, Context

# Set up constants and logging
default_log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
log_level = getattr(logging, default_log_level, logging.INFO)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.expanduser("~/.cache/llms-txt-server"))
SITES_LIST_FILE = os.path.join(CACHE_DIR, "sites.json")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
log_file = os.path.join(CACHE_DIR, "server.log")

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llms-txt-server")

# Pydantic models for data validation
class LlmsSiteEntry(BaseModel):
    domain: str
    llmsTxtUrl: str
    description: Optional[str] = None
    tokenCount: Optional[int] = None

class LlmsSitesList(BaseModel):
    sites: List[LlmsSiteEntry] = []

# Initialize or load sites list
def initialize_sites_list() -> LlmsSitesList:
    if os.path.exists(SITES_LIST_FILE):
        try:
            with open(SITES_LIST_FILE, 'r') as f:
                data = json.load(f)
                return LlmsSitesList(sites=data.get('sites', []))
        except Exception as e:
            logger.error(f"Error loading sites list: {e}")

    # Create default empty sites list if not exists
    sites_list = LlmsSitesList()
    save_sites_list(sites_list)
    return sites_list

def save_sites_list(sites_list: LlmsSitesList):
    try:
        with open(SITES_LIST_FILE, 'w') as f:
            json.dump(sites_list.model_dump(), f, indent=2)
    except Exception as e:
        logger.error(f"Error saving sites list: {e}")

# Adapter pattern for different data sources
class SiteDataAdapter(ABC):
    @abstractmethod
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from source and return as a list of dicts"""
        pass

    @abstractmethod
    def convert_to_site_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert source item to standardized site entry format"""
        pass

class KrishSiteDataAdapter(SiteDataAdapter):
    """Adapter for krish-adi/llmstxt-site data format"""
    def __init__(self, url: str = "https://raw.githubusercontent.com/krish-adi/llmstxt-site/refs/heads/main/data.json"):
        self.url = url

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            if response.status_code == 200:
                return response.json()
            return []

    def convert_to_site_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        domain = extract_domain(item["website"])
        token_count = None
        if "llms-txt-tokens" in item and item["llms-txt-tokens"] != "":
            try:
                token_count = int(item["llms-txt-tokens"])
            except (ValueError, TypeError):
                pass

        return {
            "domain": domain,
            "llmsTxtUrl": item["llms-txt"],
            "description": f"{item['product']} website",
            "tokenCount": token_count
        }

class DavidSiteDataAdapter(SiteDataAdapter):
    """Adapter for thedaviddias/llms-txt-hub data format"""
    def __init__(self, url: str = "https://raw.githubusercontent.com/thedaviddias/llms-txt-hub/refs/heads/main/data/websites.json"):
        self.url = url

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            if response.status_code == 200:
                return response.json()
            return []

    def convert_to_site_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        domain = extract_domain(item["domain"])
        return {
            "domain": domain,
            "llmsTxtUrl": item["llmsTxtUrl"],
            "description": item.get("description", f"{item['name']} website"),
            "tokenCount": None  # This source doesn't provide token counts
        }

async def fetch_and_merge_sites_data() -> List[Dict[str, Any]]:
    """Fetch data from multiple sources and merge them, removing duplicates"""
    adapters = [
        KrishSiteDataAdapter(),
        DavidSiteDataAdapter()
    ]

    all_entries = []

    for adapter in adapters:
        raw_data = await adapter.fetch_data()
        for item in raw_data:
            try:
                entry = adapter.convert_to_site_entry(item)
                all_entries.append(entry)
            except Exception as e:
                logger.error(f"Error converting item {item}: {e}")

    # Deduplicate and merge
    merged_entries = {}

    for entry in all_entries:
        domain = entry["domain"]

        if domain not in merged_entries:
            merged_entries[domain] = entry
        else:
            # If we have a duplicate, choose the entry with the richer description
            existing_desc = merged_entries[domain].get("description", "")
            new_desc = entry.get("description", "")

            if len(new_desc) > len(existing_desc):
                # Keep the token count if available
                token_count = merged_entries[domain].get("tokenCount")
                merged_entries[domain] = entry
                if token_count and not entry.get("tokenCount"):
                    merged_entries[domain]["tokenCount"] = token_count

    return list(merged_entries.values())

async def initialize_sites_list_from_sources() -> LlmsSitesList:
    """Initialize the sites list from external sources"""
    try:
        site_entries = await fetch_and_merge_sites_data()

        # Convert to LlmsSiteEntry objects
        sites = []
        for entry in site_entries:
            site = LlmsSiteEntry(
                domain=entry["domain"],
                llmsTxtUrl=entry["llmsTxtUrl"],
                description=entry.get("description", ""),
                tokenCount=entry.get("tokenCount")
            )
            sites.append(site)

        sites_list = LlmsSitesList(sites=sites)
        save_sites_list(sites_list)
        return sites_list
    except Exception as e:
        logger.error(f"Error initializing sites list from sources: {e}")
        # Fall back to empty list
        return LlmsSitesList()


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc

def get_cache_path(domain: str, file_type: str = "llms.txt") -> str:
    """Get the path where the cached file should be stored."""
    domain_dir = os.path.join(CACHE_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    return os.path.join(domain_dir, file_type)

def fetch_llmstxt_with_cache(url_or_domain: str, force_refresh: bool = False) -> Tuple[Optional[str], bool]:
    """
    Fetch llms.txt file from a website and cache it.

    Args:
        url_or_domain: URL or domain to fetch llms.txt from
        force_refresh: Whether to force a refresh of the cache

    Returns:
        Tuple of (content, from_cache)
    """
    # Determine if it's a URL or domain
    if not url_or_domain.startswith(('http://', 'https://')):
        domain = url_or_domain
        url = f"https://{domain}/llms.txt"
    else:
        url = url_or_domain
        domain = extract_domain(url)

    cache_path = get_cache_path(domain)

    # Check if cached version exists and not forcing refresh
    if os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Using cached llms.txt for {domain}")
                return content, True
        except Exception as e:
            logger.error(f"Error reading cached llms.txt: {e}")

    # Fetch from website
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            if response.status_code == 200:
                content = response.text
                # Cache the content
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Successfully fetched and cached llms.txt for {domain}")
                return content, False
            else:
                logger.warning(f"Failed to fetch llms.txt from {url}, status code: {response.status_code}")
                return None, False
    except Exception as e:
        logger.error(f"Error fetching llms.txt from {url}: {e}")
        return None, False


def convert_webpage_to_markdown(url: str, force_refresh: bool = False) -> Tuple[Optional[str], bool]:
    """
    Fetch a webpage and convert it to Markdown.

    Args:
        url: URL of the webpage to fetch
        force_refresh: Whether to force a refresh of the cache

    Returns:
        Tuple of (content, from_cache)
    """
    domain = extract_domain(url)
    cache_path = get_cache_path(domain, "fallback.md")

    # Check if cached version exists and not forcing refresh
    if os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Using cached markdown for {url}")
                return content, True
        except Exception as e:
            logger.error(f"Error reading cached markdown: {e}")

    # Fetch and convert webpage
    try:
     with httpx.Client(timeout=10) as client:
        response = client.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove scripts, styles, and other non-content elements
            for tag in soup(['script', 'style', 'iframe', 'footer']):
                tag.decompose()

            # Convert to markdown
            markdown_content = md(str(soup),
                                  default_title = True)

            # Cache the content
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"Successfully converted and cached webpage {url} to markdown")
            return markdown_content, False
        else:
            logger.warning(f"Failed to fetch webpage {url}, status code: {response.status_code}")
            return None, False
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {e}")
        return None, False

def is_domain_in_sites_list(domain: str, sites_list: LlmsSitesList) -> Optional[LlmsSiteEntry]:
    """Check if a domain is in the sites list."""
    for site in sites_list.sites:
        if site.domain == domain:
            return site
    return None

def check_llms_support(url_or_domain: str) -> Dict:
    """Check if a website supports llms.txt."""
    # Determine if it's a URL or domain
    if url_or_domain.startswith(('http://', 'https://')):
        domain = extract_domain(url_or_domain)
    else:
        domain = url_or_domain

    sites_list = initialize_sites_list()
    site_entry = is_domain_in_sites_list(domain, sites_list)

    if site_entry:
        # Domain is in known list
        return {
            "supported": True,
            "domain": domain,
            "llmsTxtUrl": site_entry.llmsTxtUrl,
            "description": site_entry.description,
            "fromKnownList": True
        }

 # Try to check if the website has llms.txt
    try:
        url = f"https://{domain}/llms.txt"
        with httpx.Client(timeout=5) as client:
            response = client.head(url)
            if response.status_code == 200:
                new_site = LlmsSiteEntry(
                    domain=domain,
                    llmsTxtUrl=url,
                    description=f"Discovered llms.txt support for {domain}"
                )
                sites_list.sites.append(new_site)
                save_sites_list(sites_list)

                return {
                    "supported": True,
                    "domain": domain,
                    "llmsTxtUrl": url,
                    "fromKnownList": False,
                    "description": f"Discovered llms.txt support for {domain}"
                }
            else:
                return {
                    "supported": False,
                    "domain": domain,
                    "fromKnownList": False
                }
    except Exception as e:
        logger.error(f"Error checking llms.txt support for {domain}: {e}")
        return {
            "supported": False,
            "domain": domain,
            "error": str(e),
            "fromKnownList": False
        }

# ------------  MCP server code ----------
mcp = FastMCP("llms-txt-server")

@mcp.resource("llmstxt://list")
def list_llms_websites() -> str:
    """List all known websites that support llms.txt."""
    sites_list = initialize_sites_list()
    sites_info = json.dumps(sites_list.model_dump(), indent=2)
    return f"Known websites with llms.txt support:\n\n{sites_info}"

# Add a new tool to refresh sites list
@mcp.tool()
async def refresh_sites_list() -> Dict:
    """Refresh the sites list from external sources"""
    try:
        sites_list = await initialize_sites_list_from_sources()
        return {
            "success": True,
            "message": f"Successfully refreshed sites list with {len(sites_list.sites)} entries"
        }
    except Exception as e:
        logger.error(f"Error refreshing sites list: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def check_llms_support_tool(url_or_domain: str) -> Dict:
    """
    Check if a website supports llms.txt.

    Args:
        url_or_domain: URL or domain to check

    Returns:
        Information about llms.txt support
    """
    return check_llms_support(url_or_domain)

@mcp.tool()
def download_llms_txt(url_or_domain: str, force_refresh: bool = False) -> Dict:
    """
    Download llms.txt from a website.

    Args:
        url_or_domain: URL or domain to download llms.txt from
        force_refresh: Whether to force a refresh of the cache

    Returns:
        Content of llms.txt or error information
    """
    result = check_llms_support(url_or_domain)

    if not result.get("supported", False):
        return {
            "success": False,
            "error": "Website does not support llms.txt",
            "domain": result.get("domain"),
            "fallbackAvailable": True
        }

    content, from_cache = fetch_llmstxt_with_cache(
        result.get("llmsTxtUrl", f"https://{result.get('domain')}/llms.txt"),
        force_refresh
    )

    if content:
        return {
            "success": True,
            "content": content,
            "domain": result.get("domain"),
            "fromCache": from_cache
        }
    else:
        return {
            "success": False,
            "error": "Failed to fetch llms.txt",
            "domain": result.get("domain"),
            "fallbackAvailable": True
        }

@mcp.tool()
def convert_page_to_md(url: str, force_refresh: bool = False) -> Dict:
    """
    Convert a webpage to Markdown.

    Args:
        url: URL of the webpage to convert
        force_refresh: Whether to force a refresh of the cache

    Returns:
        Markdown content or error information
    """
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"

    content, from_cache = convert_webpage_to_markdown(url, force_refresh)

    if content:
        return {
            "success": True,
            "content": content,
            "url": url,
            "fromCache": from_cache
        }
    else:
        return {
            "success": False,
            "error": "Failed to convert webpage to Markdown",
            "url": url
        }

@mcp.prompt()
def check_website_llmstxt(url_or_domain: str) -> str:
    """Create a prompt to check if a website supports llms.txt."""
    return f"""
I'll check if the website {url_or_domain} supports llms.txt, which is a standardized format for providing content optimized for large language models.

If it supports llms.txt, I'll retrieve that content for you. Otherwise, I'll convert the webpage to Markdown as a fallback.

Would you like me to:
1. Just check if the website supports llms.txt
2. Retrieve the llms.txt content (if available)
3. Convert the webpage to Markdown
    """

def main():
    """Run the MCP server."""
        # Initialize sites list on startup
    sites_list = initialize_sites_list()

    # If the sites list is empty, populate it from sources
    if not sites_list.sites:
        import asyncio
        logger.info("Sites list is empty, initializing from external sources")
        asyncio.run(initialize_sites_list_from_sources())
    mcp.run()

if __name__ == '__main__':
    main()
