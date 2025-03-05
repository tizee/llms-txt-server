# TODOs

## MVP

- [ ] (Resources) Get a list of urls from cached file that supports llms.txt spec and generated
```json
{
    url: "",
    desc: "",
}
```

- [ ] (Prompts) Check a given website whether supports llms.txt spec
    - If in the list of the predefined, then return the content (truncated if it's too long) of llms.txt directly
    - Try to get the llms.txt docs from the webiste, then return error or the content(truncated if it's too long)
- [ ] (Tools) Build llms.txt for a given url and store it to the generated llms.txt urls list, return the generated  llms.txt content (truncated if it's too long)
    - For example, the docs webiste of React V19 and tailwindcss currently do not support llms.xml, we could parse their webpages to capture all links along with their main content.
    - my approach: fetch the whole web page html content and use `markdownify` to convert it to markdown directly
        - optimization: record hostnames that count more than 1, use a simple mark to denote it in the beginning of the generated llms.txt

- `mcp-server-fetch` approach: 1. use `readability` to get the content of html 2. `markdownify` to convert it to markdown
```
    ret = readabilipy.simple_json.simple_json_from_html_string(
        html, use_readability=True
    )
    if not ret["content"]:
        return "<error>Page failed to be simplified from HTML</error>"
    content = markdownify.markdownify(
        ret["content"],
        heading_style=markdownify.ATX,
    )
    return content
```

### MVP use case

```
please check the python uv tool
```

