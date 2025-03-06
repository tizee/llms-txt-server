# llms-txt-server design

## PRD

Product Requirements Document (PRD)
llms-txt-server (MVP)

1. Overview

Goal
Provide an MCP Server that seamlessly retrieves and caches llms.txt files from known websites and, if unavailable, converts HTML pages to Markdown. This consolidates two common needs—llms.txt discovery and markdown conversion—under one MCP tool, allowing LLM applications to conveniently fetch relevant textual content.

Key Value Proposition
	•	Quickly check if a given domain or URL supports llms.txt without relying on external search.
	•	Efficiently download and cache llms.txt when available.
	•	Fallback to HTML→Markdown conversion if llms.txt is missing.
	•	Unified JSON dataset of known llms.txt-enabled websites for fast lookups.

2. Target Users
	•	Developers integrating large language models into documentation tools, knowledge bases, or chatbots.
	•	Teams needing a lightweight, local server to quickly fetch standardized text content.

3. Scope and Non-Goals

In Scope (MVP)
	1.	Local JSON-based site list with basic fields (domain, llmsTxtUrl, etc.).
	2.	Simple URL/domain matching for known websites.
	3.	Download and caching of llms.txt and optional llms-full.txt.
	4.	HTML→Markdown fallback for unsupported sites.
	5.	MCP Tools for “check if domain supports llms.txt,” “download llms.txt,” and “convert page to Markdown.”

Out of Scope
	1.	Advanced security features (ACL, token auth).
	2.	Complex text chunking or advanced token management.
	3.	Large-scale distributed storage solutions.

4. User Stories
	1.	As a developer, I want to query a server to see if a website has llms.txt so I can quickly load curated info into my LLM environment.
	2.	As a developer, if a website does not support llms.txt, I want a fallback that auto-converts the page to Markdown.
	3.	As a developer, I want to avoid repeated downloads of the same llms.txt file, so I can reduce bandwidth and speed up subsequent requests.

5. Core Features
	1.	Local Website List
	•	Maintains a JSON file containing known llms.txt-enabled sites.
	•	Fields include domain, llmsTxtUrl, description, and optional tokens count.
	2.	Check & Download llms.txt
	•	Checks if domain exists in the local list.
	•	If found, downloads/caches the llms.txt file.
	•	Supports optional llms-full.txt if specified.
	3.	HTML→Markdown Fallback
	•	If llms.txt is absent, fetch and convert HTML to Markdown.
	•	Caches resulting Markdown for subsequent requests.
	4.	MCP Tools
	•	check_llms_support(url_or_keyword): Returns whether llms.txt is available.
	•	download_llms_txt(url_or_domain, forceRefresh?): Fetches and caches llms.txt.
	•	convert_page_to_md(url, forceRefresh?): Converts HTML to Markdown.
	5.	Caching & Basic Logging
	•	Stores downloaded files by domain in a local directory (data_cache/<domain>).
	•	Minimal logs tracking requests, cache hits, and errors.

6. Technical Details
	•	Data Format
	•	One or more JSON files (e.g., sites.json) storing site entries as an array of objects.
	•	Caching in local folders named by domain, including llms.txt, llms-full.txt, or fallback .md files.
	•	Deployment
	•	Runs as a lightweight Python MCP Server.
	•	No external DB required in MVP; JSON plus filesystem caching is sufficient.
	•	Fallback Logic
	•	If llmsTxtUrl lookup fails, proceed to an HTML download and Markdown conversion step.
	•	If a domain is unknown, skip directly to HTML→Markdown.
	•	Token Counting (Optional)
	•	Post-download step that uses a Python library (like tiktoken) to estimate token count.
	•	If computed, stored in the site entry and/or a metadata file.

7. Implementation Plan
	1.	Parsing JSON Site List
	•	Load data on server startup.
	•	Provide a method to refresh the list if edited.
	2.	MCP Tools Setup
	•	Implement check_llms_support, download_llms_txt, convert_page_to_md.
	3.	HTML→Markdown Utility
	•	Use an off-the-shelf converter.
	•	Basic error handling (404, redirects).
	4.	Caching
	•	On each successful download, store file and optional metadata (e.g., timestamp).
	•	Force-refresh triggers re-download.
	5.	Testing
	•	Verify with multiple known and unknown domains.
	•	Check logging outputs and cache folder contents.

8. Roadmap
	•	V1 (MVP)
	•	JSON site list, domain matching, caching, fallback conversion.
	•	V1.1+
	•	Add search by keyword in the site’s description or category.
	•	Add incremental updates to the site list.
	•	Future
	•	Optional user auth or ACL.
	•	Multi-threaded downloads and more robust error handling.

9. Success Criteria
	•	Functional: LLMs can call the tools to retrieve llms.txt or fallback Markdown for at least 10 test sites without errors.
	•	Performance: Repeated calls on the same domain leverage cache effectively.
	•	Adoption: Early adopters integrate it into their dev workflows (e.g., doc-assistant chatbots).

End of PRD
