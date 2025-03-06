# llms-txt-server

A Model Context Protocol (MCP) server for fetching and managing llms.txt files from websites for Large Language Models.

## Overview

The llms-txt-server provides a standardized interface for LLMs to interact with websites that support the llms.txt protocol.

The llms.txt format is a standardized way for websites to provide content optimized for large language models, similar to robots.txt for web crawlers. This server helps LLMs discover, fetch, and process llms.txt files from websites, with fallback to webpage conversion when llms.txt isn't available.

## Features

- ✅ Check if a website supports llms.txt
- 📋 List known websites with llms.txt support
- 🔄 Refresh sites list from external sources
- 📥 Fetch llms.txt content with caching
- 📄 Convert webpages to Markdown when llms.txt isn't available
- 🔌 Full MCP integration with Claude Desktop

## Installation

### Prerequisites

- Python 3.11 or higher

### Install from source

```bash
# Clone the repository
git clone  llms-txt-server
cd llms-txt-server

# Install the package
pip install -e .
```

## Usage

### Running the server

```bash
# Start the server directly
llms-txt-server

# Or via the MCP CLI
mcp run src/llms_txt_server/server.py
```

### Installing in Claude Desktop

```bash
mcp install src/llms_txt_server/server.py
```


## Available Tools

| Tool | Description |
|------|-------------|
| `list_llms_websites` | List all known websites that support llms.txt |
| `refresh_sites_list` | Refresh the sites list from external sources |
| `check_llms_support_tool` | Check if a website supports llms.txt |
| `get_llms_txt` | Get llms.txt content from a website |
| `convert_page_to_md` | Convert a webpage to Markdown |

## Prompts

| Prompt | Description |
|--------|-------------|
| `fetch_url` | Create a prompt to fetch a URL that prefers domain's llms.txt if available |

## How it Works

1. The server maintains a cached list of websites known to support llms.txt
2. When a URL is requested, it first checks if the site supports llms.txt
3. If supported, it fetches the llms.txt content (with caching)
4. If not supported, it falls back to converting the webpage to Markdown
5. The server periodically refreshes its list from external sources

## Configuration

You can configure the server using environment variables:

- `LOG_LEVEL`: Set logging level (default: "DEBUG")
- `CACHE_DIR`: Set cache directory (default: "~/.cache/llms-txt-server")

## Development

### Setup Development Environment

```bash
npx @modelcontextprotocol/inspector uv run llms-txt-server
```

### Code Style

This project uses Ruff for formatting and linting. See `pyproject.toml` for configuration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
