import json
import os
import pytest
import tempfile
from unittest import mock

# Import from the package
from llms_txt_server.server import (
    extract_domain,
    get_cache_path,
    fetch_llmstxt_with_cache,
    convert_webpage_to_markdown,
    check_llms_support,
    initialize_sites_list,
    save_sites_list,
    LlmsSitesList,
    LlmsSiteEntry
)

# Mock data for testing
MOCK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <script>console.log('This should be removed');</script>
    <style>body { color: red; }</style>
</head>
<body>
    <h1>Test Page</h1>
    <p>This is a test paragraph.</p>
    <nav>This navigation should be removed</nav>
</body>
</html>
"""

MOCK_LLMS_TXT = """
# Example llms.txt
This is a test llms.txt file used for unit tests.
It contains text formatted for consumption by LLMs.
"""

MOCK_SITES_LIST = {
    "sites": [
        {
            "domain": "example.com",
            "llmsTxtUrl": "https://example.com/llms.txt",
            "description": "Example website with llms.txt"
        }
    ]
}

# Setup temporary directory for tests
@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch('llms_txt_server.server.CACHE_DIR', tmpdir):
            with mock.patch('llms_txt_server.server.SITES_LIST_FILE', os.path.join(tmpdir, 'sites.json')):
                yield tmpdir

# Test URL parsing and domain extraction
def test_extract_domain():
    assert extract_domain("https://example.com/path") == "example.com"
    assert extract_domain("http://subdomain.example.com") == "subdomain.example.com"
    assert extract_domain("https://example.com") == "example.com"

# Test cache path generation
def test_get_cache_path(temp_cache_dir):
    domain = "example.com"
    expected_path = os.path.join(temp_cache_dir, domain, "llms.txt")
    assert get_cache_path(domain) == expected_path

# Test sites list initialization and saving
def test_initialize_and_save_sites_list(temp_cache_dir):
    # Test initialization with no existing file
    sites_list = initialize_sites_list()
    assert isinstance(sites_list, LlmsSitesList)
    assert len(sites_list.sites) == 0

    # Test saving
    test_site = LlmsSiteEntry(
        domain="test.com",
        llmsTxtUrl="https://test.com/llms.txt",
        description="Test site"
    )
    sites_list.sites.append(test_site)
    save_sites_list(sites_list)

    # Test loading saved list
    loaded_list = initialize_sites_list()
    assert len(loaded_list.sites) == 1
    assert loaded_list.sites[0].domain == "test.com"

# Test fetching llms.txt with mocked responses
def test_fetch_llmstxt_with_cache(temp_cache_dir):
    domain = "example.com"
    _= f"https://{domain}/llms.txt"

    # Mock successful response using httpx
    with mock.patch('httpx.Client') as mock_client:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_LLMS_TXT
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Test fetching from website
        content, from_cache = fetch_llmstxt_with_cache(domain)
        assert content == MOCK_LLMS_TXT
        assert from_cache is False

    # Test fetching from cache
    with mock.patch('httpx.Client') as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Should not reach here")
        content, from_cache = fetch_llmstxt_with_cache(domain)
        assert content == MOCK_LLMS_TXT
        assert from_cache is True

    # Test force refresh
    with mock.patch('httpx.Client') as mock_client:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_LLMS_TXT + "\nUpdated"
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        content, from_cache = fetch_llmstxt_with_cache(domain, force_refresh=True)
        assert content == MOCK_LLMS_TXT + "\nUpdated"
        assert from_cache is False

# Test converting webpage to markdown
def test_convert_webpage_to_markdown(temp_cache_dir):
    url = "https://example.com"

    # Mock successful response using httpx
    with mock.patch('httpx.Client') as mock_client:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_HTML
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Test converting from website
        content, from_cache = convert_webpage_to_markdown(url)
        assert content is not None
        assert "Test Page" in content
        assert "This is a test paragraph." in content
        assert "console.log" not in content
        assert from_cache is False

    # Test fetching from cache
    with mock.patch('httpx.Client') as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Should not reach here")
        content, from_cache = convert_webpage_to_markdown(url)
        assert content is not None
        assert "Test Page" in content
        assert from_cache is True

# Test checking if a website supports llms.txt
def test_check_llms_support(temp_cache_dir):
    domain = "example.com"

    # Setup sites list with known domain
    with open(os.path.join(temp_cache_dir, "sites.json"), "w") as f:
        json.dump(MOCK_SITES_LIST, f)

    # Test with known domain
    result = check_llms_support(domain)
    assert result["supported"] is True
    assert result["domain"] == domain
    assert result["fromKnownList"] is True

    # Test with unknown domain that has llms.txt
    with mock.patch('httpx.Client') as mock_client:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.head.return_value = mock_response

        result = check_llms_support("unknown.com")
        assert result["supported"] is True
        assert result["domain"] == "unknown.com"
        assert result["fromKnownList"] is False

    # Test with domain that doesn't have llms.txt
    with mock.patch('httpx.Client') as mock_client:
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_client.return_value.__enter__.return_value.head.return_value = mock_response

        result = check_llms_support("no-llms.com")
        assert result["supported"] is False
        assert result["domain"] == "no-llms.com"
        assert result["fromKnownList"] is False
