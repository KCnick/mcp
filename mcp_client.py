import asyncio
import json
from typing import List, Optional
from langchain.tools import StructuredTool
from pydantic import Field
from pydantic.v1 import BaseModel as V1BaseModel, Field as V1Field
import httpx


class MCPHTTPClient:
    """HTTP client for MCP server communication using SSE"""
    
    def __init__(self, server_url: str, server_name: str):
        self.server_url = server_url.rstrip('/')
        self.server_name = server_name
        self.tools_cache = None
        self.message_id = 1
        
    def _get_next_id(self) -> int:
        """Get next message ID"""
        msg_id = self.message_id
        self.message_id += 1
        return msg_id
        
    async def _make_sse_request(self, method: str, params: Optional[dict] = None) -> dict:
        """Make SSE request to MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                request_data = {
                    "jsonrpc": "2.0",
                    "id": self._get_next_id(),
                    "method": method,
                    "params": params or {}
                }
                
                # Try POST with SSE
                try:
                    async with client.stream(
                        "POST",
                        self.server_url,
                        json=request_data,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive"
                        }
                    ) as response:
                        response.raise_for_status()
                        
                        # Read SSE events
                        result_data = None
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str.strip() and data_str != "[DONE]":
                                    try:
                                        event_data = json.loads(data_str)
                                        if "result" in event_data:
                                            result_data = event_data["result"]
                                        elif "error" in event_data:
                                            raise Exception(f"MCP Error: {event_data['error']}")
                                    except json.JSONDecodeError:
                                        continue
                        
                        if result_data is None:
                            return {}
                        
                        return result_data
                        
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 406:
                        # Try regular JSON POST instead
                        print(f"SSE not supported, trying regular JSON POST...")
                        response = await client.post(
                            self.server_url,
                            json=request_data,
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "application/json"
                            }
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        if "error" in data:
                            raise Exception(f"MCP Error: {data['error']}")
                        
                        return data.get("result", {})
                    else:
                        raise
                    
            except httpx.HTTPStatusError as e:
                print(f"HTTP error for {method}: {e.response.status_code}")
                try:
                    error_text = await e.response.aread()
                    print(f"Response: {error_text.decode()}")
                except:
                    print("Could not read error response")
                raise
            except Exception as e:
                print(f"Error making MCP request to {method}: {e}")
                raise
    
    async def initialize(self) -> dict:
        """Initialize connection with MCP server"""
        return await self._make_sse_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "support-chatbot",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -> List[dict]:
        """Get list of available tools from MCP server"""
        if self.tools_cache:
            return self.tools_cache
            
        result = await self._make_sse_request("tools/list")
        self.tools_cache = result.get("tools", [])
        return self.tools_cache
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a specific tool on the MCP server"""
        result = await self._make_sse_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        # Extract content from result
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", str(result))
        return str(result)


class MultiServerMCPClient:
    """Client for managing multiple MCP servers"""
    
    def __init__(self, servers_config: dict):
        """
        Initialize with server configurations
        
        Args:
            servers_config: Dict with server names as keys and config dicts as values
                           e.g., {"aws-service": {"url": "...", "transport": "http"}}
        """
        self.servers_config = servers_config
        self.sessions = {}
        self.tools_cache = None
        
    async def _connect_http_server(self, name: str, config: dict):
        """Connect to an HTTP-based MCP server"""
        client = MCPHTTPClient(config["url"], name)
        await client.initialize()
        return client
    
    async def initialize(self):
        """Initialize connections to all configured servers"""
        for server_name, config in self.servers_config.items():
            transport = config.get("transport", "http")
            
            if transport == "http":
                session = await self._connect_http_server(server_name, config)
                self.sessions[server_name] = session
            else:
                raise ValueError(f"Unsupported transport type: {transport}")
    
    async def get_tools(self) -> List[dict]:
        """Get all tools from all connected servers"""
        if self.tools_cache:
            return self.tools_cache
        
        all_tools = []
        for server_name, session in self.sessions.items():
            tools = await session.list_tools()
            # Add server name to each tool for tracking
            for tool in tools:
                tool["_server"] = server_name
            all_tools.extend(tools)
        
        self.tools_cache = all_tools
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: dict, server_name: str = None) -> str:
        """Call a tool on the appropriate server"""
        if server_name and server_name in self.sessions:
            session = self.sessions[server_name]
        else:
            # Find the server that has this tool
            session = None
            for name, sess in self.sessions.items():
                tools = await sess.list_tools()
                if any(t.get("name") == tool_name for t in tools):
                    session = sess
                    break
            
            if not session:
                raise ValueError(f"Tool {tool_name} not found on any server")
        
        return await session.call_tool(tool_name, arguments)


class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain tools"""
    
    def __init__(self, mcp_client: MultiServerMCPClient):
        self.mcp_client = mcp_client
        
    async def get_langchain_tools(self) -> List[StructuredTool]:
        """Convert MCP tools to LangChain tools"""
        mcp_tools = await self.mcp_client.get_tools()
        langchain_tools = []
        
        for tool in mcp_tools:
            lc_tool = self._convert_to_langchain_tool(tool)
            if lc_tool:
                langchain_tools.append(lc_tool)
                
        return langchain_tools
    
    def _convert_to_langchain_tool(self, mcp_tool: dict) -> Optional[StructuredTool]:
        """Convert a single MCP tool to LangChain StructuredTool"""
        try:
            tool_name = mcp_tool.get("name", "")
            description = mcp_tool.get("description", "")
            input_schema = mcp_tool.get("inputSchema", {})
            server_name = mcp_tool.get("_server")
            
            # Create dynamic Pydantic v1 model for input
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            # Build fields dict for Pydantic v1 model with proper typing
            field_definitions = {}
            annotations = {}
            
            for prop_name, prop_info in properties.items():
                # Determine field type
                field_type = str
                prop_type = prop_info.get("type", "string")
                
                if prop_type == "number":
                    field_type = float
                elif prop_type == "integer":
                    field_type = int
                elif prop_type == "boolean":
                    field_type = bool
                elif prop_type == "array":
                    field_type = list
                elif prop_type == "object":
                    field_type = dict
                
                # Handle required vs optional fields
                is_required = prop_name in required
                
                if is_required:
                    annotations[prop_name] = field_type
                    field_definitions[prop_name] = V1Field(
                        ..., 
                        description=prop_info.get("description", "")
                    )
                else:
                    from typing import Optional as Opt
                    annotations[prop_name] = Opt[field_type]
                    field_definitions[prop_name] = V1Field(
                        default=None,
                        description=prop_info.get("description", "")
                    )
            
            # Create dynamic Pydantic v1 model properly
            InputModel = type(
                f"{tool_name.replace('-', '_').replace('.', '_')}_Input",
                (V1BaseModel,),
                {
                    "__annotations__": annotations,
                    **field_definitions
                }
            )
            
            # Create async function to call tool
            async def call_tool(**kwargs):
                # Filter out None values for optional parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return await self.mcp_client.call_tool(tool_name, filtered_kwargs, server_name)
            
            # Wrap async function for sync LangChain
            def sync_call_tool(**kwargs):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(call_tool(**kwargs))
                finally:
                    loop.close()
            
            return StructuredTool(
                name=tool_name,
                description=description,
                func=sync_call_tool,
                args_schema=InputModel
            )
            
        except Exception as e:
            import traceback
            print(f"Error converting tool {mcp_tool.get('name')}: {e}")
            print(traceback.format_exc())
            return None


async def setup_mcp_client(server_url: str, server_name: str) -> tuple[MultiServerMCPClient, List[StructuredTool]]:
    """Setup MCP client and return LangChain tools"""
    # Create client with server configuration
    client = MultiServerMCPClient({
        server_name: {
            "url": server_url,
            "transport": "http"
        }
    })
    
    # Initialize connection
    print(f"Connecting to {server_name} and fetching tool definitions...")
    await client.initialize()
    
    # Get tools
    wrapper = MCPToolWrapper(client)
    tools = await wrapper.get_langchain_tools()
    
    print(f"Successfully loaded {len(tools)} tools from {server_name}\n")
    
    return client, tools