import asyncio
import json
from fastmcp import Client
from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from typing import Dict, Any, Tuple
import re

# Load environment variables
load_dotenv()

# Configuration
MCP_SERVER_URL = "http://localhost:8000/mcp"
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDAxMjBAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.zs2nvUrUehlJ4zd5xD9QDAXGyh-s55O9x2xUY0gxNwA")

class MCPChatClient:
    """MCP Client with OpenAI integration for natural language interaction"""
    
    def __init__(self, server_url: str, openai_api_key: str):
        self.server_url = server_url
        self.openai_client = OpenAI(
    base_url="https://aipipe.org/openrouter/v1",
    api_key=AIPIPE_TOKEN

)
        self.conversation_history: List[Dict[str, str]] = []
        self.available_tools = []
        self.available_resources = []
        self.available_prompts = []
        
    async def initialize(self):
        """Initialize connection and fetch available tools/resources"""
        async with Client(self.server_url) as client:
            # Get available tools
            tools_response = await client.list_tools()
            # Handle both list and object responses
            tools_list = tools_response if isinstance(tools_response, list) else tools_response.tools
            self.available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                for tool in tools_list
            ]
            
            # Get available resources
            resources_response = await client.list_resources()
            resources_list = resources_response if isinstance(resources_response, list) else resources_response.resources
            self.available_resources = [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description
                }
                for resource in resources_list
            ]
            
            # Get available prompts
            try:
                prompts_response = await client.list_prompts()
                prompts_list = prompts_response if isinstance(prompts_response, list) else prompts_response.prompts
                self.available_prompts = [
                    {
                        "name": prompt.name,
                        "description": prompt.description
                    }
                    for prompt in prompts_list
                ]
            except Exception as e:
                print(f"⚠️  Could not fetch prompts: {e}")
                self.available_prompts = []
            
        print("✅ Connected to MCP Server")
        print(f"📦 Available tools: {len(self.available_tools)}")
        print(f"📚 Available resources: {len(self.available_resources)}")
        print(f"💬 Available prompts: {len(self.available_prompts)}")
        
    def _build_system_prompt(self) -> str:
        """Build system prompt with available MCP capabilities"""
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.available_tools
        ])
        
        resources_desc = "\n".join([
            f"- {res['uri']}: {res['description']}"
            for res in self.available_resources
        ])
        
        return f"""You are an AI assistant with access to a Model Context Protocol (MCP) server for user management.

Available Tools (use these to perform actions):
{tools_desc}

Available Resources (use these to read data):
{resources_desc}

When the user asks you to do something:
1. Analyze what they want
2. Decide if you need to use a tool or read a resource
3. Use the appropriate MCP capability
4. Provide a natural, helpful response 

### IMPORTANT: ALWAYS FOLLOW
If you have sufficient data required for required tool / resource to call then only reply in tool call JSON or resource JSON.

For tool calls, respond with JSON in this format:
{{"action": "call_tool", "tool_name": "tool-name", "arguments": {{"key": "value"}}}}

For resource reads, respond with JSON in this format:
{{"action": "read_resource", "uri": "resource-uri"}}

"""

    async def _execute_mcp_action(self, action: Dict[str, Any]) -> str:
        """Execute an MCP tool call or resource read"""
        async with Client(self.server_url) as client:
            if action["action"] == "call_tool":
                print(f"🔧 Calling tool: {action['tool_name']}")
                result = await client.call_tool(
                    action["tool_name"],
                    action.get("arguments", {})
                )
                return result.content[0].text
                
            elif action["action"] == "read_resource":
                print(f"📖 Reading resource: {action['uri']}")
                result = await client.read_resource(action["uri"])
                return result[0].text
                
        return "Error: Unknown action type"

    def _parse_llm_response(self, response: str) -> Tuple[str | None, Dict[str, Any] | None]:
        """Parse LLM response to check if it contains an MCP action (JSON block)."""
        try:
            # Try to find a JSON object inside any text
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match:
                json_part = match.group(0)
                action = json.loads(json_part)
                print("✅ Extracted JSON action:", action)

                if "action" not in action:
                    raise ValueError("Parsed JSON does not contain required key 'action'.")

                return None, action

            # No JSON block found — treat as plain text
            return response.strip(), None

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e.msg} | Extracted: {json_part!r}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while parsing LLM response: {e}") from e

    
    async def chat(self, user_message: str) -> str:
        """Process a user message with LLM and MCP integration"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + self.conversation_history
        
        try:
            # Get LLM response
            response = self.openai_client.chat.completions.create(
                model="openai/gpt-4o",  # or "gpt-4"
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse response
            text_response, mcp_action = self._parse_llm_response(llm_response)
            
            # If there's an MCP action, execute it
            if mcp_action:
                mcp_result = await self._execute_mcp_action(mcp_action)
                
                # Send result back to LLM for a natural response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": llm_response
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"MCP Result: {mcp_result}\n\nPlease provide a natural response to the user based on this result."
                })
                
                # Get final response from LLM
                final_response = self.openai_client.chat.completions.create(
                    model="openai/gpt-4o",
                    messages=[
                        {"role": "system", "content": self._build_system_prompt()}
                    ] + self.conversation_history,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_text = final_response.choices[0].message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })
                
                return final_text
            else:
                # Regular text response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": text_response
                })
                return text_response
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared")


async def main():
    """Main interactive chat loop"""
    print("\n" + "="*60)
    print("🤖 MCP CHAT CLIENT WITH OPENAI")
    print("="*60)
    print("Type your queries naturally. Examples:")
    print("  - 'Show me all users'")
    print("  - 'Create a new user named John with email john@example.com'")
    print("  - 'Get details for user with ID 1'")
    print("  - 'Create an admin user'")
    print("\nType 'exit' to quit, 'reset' to clear history")
    print("="*60 + "\n")
    
    # Initialize client
    client = MCPChatClient(MCP_SERVER_URL, AIPIPE_TOKEN)
    
    try:
        await client.initialize()
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        print("Make sure your MCP server is running at", MCP_SERVER_URL)
        return
    
    print("\n✨ Ready to chat!\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "exit":
                print("\n👋 Goodbye!")
                break
                
            if user_input.lower() == "reset":
                client.reset_conversation()
                continue
            
            print()  # Blank line for readability
            
            # Get response
            response = await client.chat(user_input)
            
            print(f"\n🤖 Assistant: {response}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# Alternative: Batch query processing
async def batch_queries():
    """Process multiple queries in batch"""
    client = MCPChatClient(MCP_SERVER_URL, AIPIPE_TOKEN)
    await client.initialize()
    
    queries = [
        "Show me all users in the database",
        "Create a new user named Alice with email alice@example.com, address 123 Main St, and phone 555-1234",
        "Get the details for user ID 1",
        "Create another user named Bob with email bob@test.com, address 456 Oak Ave, phone 555-5678"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        response = await client.chat(query)
        print(f"\n🤖 Response: {response}\n")
        
        await asyncio.sleep(1)  # Rate limiting


if __name__ == "__main__":
    # Interactive mode
    asyncio.run(main())
    
    # Or batch mode:
    # asyncio.run(batch_queries())