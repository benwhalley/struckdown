# MCP (Model Context Protocol) Support for Struckdown Actions

This document outlines how struckdown's action system could be extended to call tools exposed by MCP servers.

---

## Background

MCP (Model Context Protocol) is an open standard for connecting LLM applications to external data sources and tools. MCP servers expose "tools" -- callable functions with JSON Schema parameters -- that clients can invoke.

Struckdown already has an action system (`[[@action:var|params]]`) that executes Python functions during template processing. Extending this to call MCP tools would let struckdown templates access external capabilities without writing custom Python code.

---

## Use Cases

1. **RAG pipelines**: Fetch documents from a file server, database, or knowledge base
2. **Data enrichment**: Look up information from external APIs during extraction
3. **Multi-system orchestration**: Coordinate between different MCP-enabled services
4. **Reusing existing tools**: Leverage the growing ecosystem of MCP servers (GitHub, Slack, databases, etc.)

---

## Proposed Design

### Option A: Generic `@mcp` Action

A single action that takes server and tool as parameters:

```markdown
[[@mcp:result|server="filesystem", tool="read_file", path="/data/report.txt"]]

[[@mcp:commits|server="github", tool="list_commits", repo="owner/repo", limit=10]]
```

**Pros:**
- Simple implementation
- Explicit about what's being called
- No namespace collisions

**Cons:**
- Verbose syntax
- No auto-completion or validation of tool parameters

### Option B: Auto-Register MCP Tools as Actions

On startup, connect to configured MCP servers and register each tool as a struckdown action:

```markdown
[[@github_list_commits:commits|repo="owner/repo", limit=10]]

[[@filesystem_read_file:content|path="/data/report.txt"]]
```

**Pros:**
- Cleaner template syntax
- Could validate parameters against tool schemas
- Feels native to struckdown

**Cons:**
- Potential namespace collisions (need prefixing strategy)
- Startup cost to discover tools
- More complex implementation

### Recommended: Hybrid Approach

1. Support explicit `@mcp` action for flexibility
2. Allow opt-in auto-registration with configurable prefixes
3. Validate parameters against MCP tool schemas where possible

---

## Implementation Plan

### Phase 1: Core MCP Client

Create `struckdown/mcp/` module:

```
struckdown/mcp/
    __init__.py       # exports
    client.py         # MCP client wrapper
    transport.py      # stdio/SSE transport implementations
    registry.py       # server configuration and tool discovery
```

**Dependencies:**
- `mcp` -- official MCP Python SDK (if stable), or implement minimal client
- `anyio` -- async transport handling

**Key classes:**

```python
@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str                    # server identifier (e.g., "github")
    command: list[str] | None    # for stdio transport: ["npx", "-y", "@mcp/github"]
    url: str | None              # for SSE transport: "http://localhost:3000/sse"
    env: dict[str, str] = field(default_factory=dict)


class MCPRegistry:
    """Manages MCP server connections and tool discovery."""

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPTool] = {}  # "server.tool" -> tool

    def add_server(self, config: MCPServerConfig) -> None: ...
    def connect(self, server_name: str) -> None: ...
    def disconnect(self, server_name: str) -> None: ...
    def list_tools(self, server_name: str | None = None) -> list[MCPTool]: ...
    def call_tool(self, server: str, tool: str, arguments: dict) -> Any: ...
```

### Phase 2: Generic @mcp Action

```python
# struckdown/actions/mcp_action.py

from struckdown.actions import Actions
from struckdown.mcp import get_registry

@Actions.register("mcp", on_error="propagate", default_save=True)
async def mcp_action(
    context: dict,
    server: str,
    tool: str,
    **kwargs
) -> str:
    """Call an MCP tool and return result.

    Usage:
        [[@mcp:result|server="github", tool="list_commits", repo="owner/repo"]]

    Args:
        context: Struckdown context
        server: MCP server name (as configured)
        tool: Tool name to invoke
        **kwargs: Tool arguments

    Returns:
        Tool result as string (JSON for complex results)
    """
    registry = get_registry()
    result = await registry.call_tool(server, tool, kwargs)

    # serialise result for template use
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    return str(result)
```

### Phase 3: Configuration

Support multiple configuration methods:

**1. Environment variables:**
```bash
STRUCKDOWN_MCP_SERVERS='[{"name":"github","command":["npx","-y","@modelcontextprotocol/server-github"]}]'
```

**2. Configuration file (`struckdown.toml` or `pyproject.toml`):**
```toml
[tool.struckdown.mcp.servers.github]
command = ["npx", "-y", "@modelcontextprotocol/server-github"]

[tool.struckdown.mcp.servers.filesystem]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]

[tool.struckdown.mcp.servers.custom]
url = "http://localhost:3000/sse"
```

**3. Programmatic configuration:**
```python
from struckdown.mcp import MCPRegistry, MCPServerConfig

registry = MCPRegistry()
registry.add_server(MCPServerConfig(
    name="github",
    command=["npx", "-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]}
))
```

### Phase 4: Auto-Registration (Optional)

```python
# struckdown/mcp/autoregister.py

def register_mcp_tools(
    registry: MCPRegistry,
    prefix_style: Literal["server_tool", "tool_only"] = "server_tool",
    servers: list[str] | None = None,
) -> list[str]:
    """Auto-register MCP tools as struckdown actions.

    Args:
        registry: MCP registry with connected servers
        prefix_style: How to name actions:
            - "server_tool": @github_list_commits
            - "tool_only": @list_commits (may collide)
        servers: Specific servers to register (None = all)

    Returns:
        List of registered action names
    """
    registered = []

    for tool in registry.list_tools(servers):
        if prefix_style == "server_tool":
            action_name = f"{tool.server}_{tool.name}"
        else:
            action_name = tool.name

        # create action dynamically
        @Actions.register(action_name, on_error="propagate")
        async def tool_action(context: dict, _tool=tool, **kwargs):
            result = await registry.call_tool(_tool.server, _tool.name, kwargs)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)

        registered.append(action_name)

    return registered
```

### Phase 5: CLI Integration

```bash
# list available MCP tools
sd mcp list

# call a tool directly
sd mcp call github list_commits --repo owner/repo

# run template with MCP servers
sd chat -p template.sd --mcp-server github --mcp-server filesystem
```

---

## Template Syntax Examples

### Basic tool call:
```markdown
[[@mcp:readme|server="filesystem", tool="read_file", path="README.md"]]

Based on the README:
{{readme}}

Summarise in one paragraph: [[summary]]
```

### With auto-registered tools:
```markdown
[[@github_list_issues:issues|repo="owner/repo", state="open"]]

Open issues:
{{issues}}

Categorise by priority: [[json:categorised]]
```

### Chained calls:
```markdown
[[@mcp:files|server="filesystem", tool="list_directory", path="/docs"]]

<checkpoint>

{% for file in files %}
[[@mcp:content_{{loop.index}}|server="filesystem", tool="read_file", path="{{file}}"]]
{% endfor %}

<checkpoint>

Summarise all documents: [[summary]]
```

---

## Error Handling

MCP tool calls can fail in several ways:

1. **Server not connected** -- raise `MCPServerNotConnectedError`
2. **Tool not found** -- raise `MCPToolNotFoundError`
3. **Invalid arguments** -- validate against tool schema, raise `MCPArgumentError`
4. **Tool execution error** -- propagate or handle per `on_error` setting

```python
class MCPError(StruckdownError):
    """Base class for MCP errors."""

class MCPServerNotConnectedError(MCPError):
    """Server is not connected."""

class MCPToolNotFoundError(MCPError):
    """Tool not found on server."""

class MCPArgumentError(MCPError):
    """Invalid arguments for tool."""

class MCPExecutionError(MCPError):
    """Tool execution failed."""
```

---

## Security Considerations

1. **Remote use**: MCP actions should have `allow_remote_use=False` by default (like `@fetch`)
2. **Server sandboxing**: Document that MCP servers run with their own permissions
3. **Credential handling**: Never log or expose credentials passed via env vars
4. **Tool allowlists**: Consider supporting explicit allowlists per server

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=0.1.0",     # official SDK when stable
    "anyio>=4.0.0",   # async transport
]
```

---

## Open Questions

1. **Lifecycle management**: Should MCP servers be started/stopped automatically, or managed externally?
2. **Connection pooling**: Keep connections open between calls, or connect per-call?
3. **Tool schema validation**: Validate parameters before calling, or let server validate?
4. **Result type coercion**: How to handle complex tool results (nested JSON, binary data)?
5. **Streaming**: Support for streaming tool results (e.g., large file reads)?

---

## Not In Scope

This extension makes struckdown a **consumer** of MCP tools. It does NOT:

- Turn struckdown into an MCP server (exposing its own tools)
- Support agentic tool-calling loops (LLM decides which tool to call)
- Replace the existing action system

These could be separate future extensions if needed.
