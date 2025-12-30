import asyncio
import os
from dotenv import load_dotenv
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.human_input.console_handler import console_input_callback
from mcp_agent.elicitation.handler import console_elicitation_callback

# Load environment variables
load_dotenv()

# Configuration
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "xxxx")

# Initialize MCP App
app = MCPApp(
    name="user_management_chat_agent",
    human_input_callback=console_input_callback,
    elicitation_callback=console_elicitation_callback,
)

class MCPChatAgent:
    """MCP Agent-based chat client for natural language interaction"""
    
    def __init__(self):
        self.agent = None
        self.llm = None
        self.agent_app = None
        
    async def initialize(self, mcp_agent_app):
        """Initialize the agent and attach LLM"""
        self.agent_app = mcp_agent_app
        logger = self.agent_app.logger

        self.agent = Agent(
            name="user_manager",
            instruction="""
            You are a helpful assistant for user management.

            When the user asks you to do something:
            1. Analyze what they want
            2. Use the appropriate MCP tools/resources to fulfill the request
            3. Provide a natural, helpful response 
            
            Available capabilities:
            - Use tools for operations (create, read, update, delete users)
            - Read resources for documentation and reference data
            - Use prompts for structured interactions
            
            If you cannot find a suitable tool, explain what's available and ask for clarification.
            Always be helpful and guide the user on what you can do.
            """,
            server_names=["user_management"],
        )
        
        await self.agent.__aenter__()  # Keep agent session open
        
        # List available tools
        tools_result = await self.agent.list_tools()
        tools = tools_result.tools

        print(f"\n✅ Connected to MCP Server")
        print(f"📦 Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description or 'No description'}")

        # List resources
        try:
            resources_result = await self.agent.list_resources("user_management")
            resources = resources_result.resources
            print(f"\n📚 Available resources: {len(resources)}")
            for resource in resources:
                print(f"  - {resource.uri}: {resource.name or 'No name'}")
        except Exception as e:
            logger.warning(f"No resources available: {e}")

        # List prompts
        try:
            prompts_result = await self.agent.list_prompts("user_management")
            prompts = prompts_result.prompts
            print(f"\n💬 Available prompts: {len(prompts)}")
            for prompt in prompts:
                print(f"  - {prompt.name}: {prompt.description or 'No description'}")
        except Exception as e:
            logger.warning(f"No prompts available: {e}")

        # Attach OpenAI LLM
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

        print("\n✨ Ready to chat!\n")
            
    async def chat(self, user_message: str, auto_include_resources: bool = False) -> str:
        """
        Process a user message and return response
        
        Args:
            user_message: The user's query
            auto_include_resources: If True, automatically include all available resources as context
        """
        try:
            if auto_include_resources:
                # Approach 1: Include ALL resources automatically
                messages = [user_message]
                
                try:
                    resources_result = await self.agent.list_resources("user_management")
                    for resource in resources_result.resources:
                        try:
                            content = await self.agent.read_resource(
                                uri=resource.uri,
                                server_name="user_management"
                            )
                            messages.append(f"Available resource - {resource.name} ({resource.uri}):\n{content}")
                        except Exception as e:
                            self.agent_app.logger.warning(f"Failed to read resource {resource.uri}: {e}")
                except Exception as e:
                    self.agent_app.logger.warning(f"No resources to include: {e}")
                
                response = await self.llm.generate_str(message=messages)
            else:
                # Default: Only tools are available, no resources
                # The LLM will automatically:
                # 1. Analyze the user's request
                # 2. Decide which tools to call
                # 3. Execute the tools
                # 4. Synthesize a natural response
                response = await self.llm.generate_str(message=user_message)
            
            return response
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    async def chat_with_llm_resource_selection(self, user_message: str) -> str:
        """
        APPROACH 3: Let the LLM decide which resources to use
        First call LLM to select resources, then call again with selected resources
        """

        print("Inside chat with llm resources")
        try:
            # Step 1: List available resources
            resources_list = []
            try:
                resources_result = await self.agent.list_resources("user_management")
                resources_list = [
                    f"- {r.uri}: {r.name or 'No description'}" 
                    for r in resources_result.resources
                ]
            except:
                pass
            
            if not resources_list:
                # No resources, just do normal chat
                return await self.chat(user_message)
            
            # Step 2: Ask LLM which resources are needed
            print("preparing prompt for selected resource")
            
            resource_selection_prompt = f"""
Available resources:
{chr(10).join(resources_list)}

User query: {user_message}

Which resources (if any) would be helpful to answer this query? 
Respond with ONLY the URIs, one per line, or "NONE" if no resources are needed.
"""
            
            selected_uris_response = await self.llm.generate_str(message=resource_selection_prompt)
            
            # Step 3: Parse the response and get selected resources
            messages = [user_message]
            
            if "NONE" not in selected_uris_response.upper():
                selected_uris = [uri.strip() for uri in selected_uris_response.split('\n') if uri.strip()]
                
                for uri in selected_uris:
                    try:
                        content = await self.agent.read_resource(
                            uri=uri,
                            server_name="user_management"
                        )
                        messages.append(f"Resource context from {uri}:\n{content}")
                        self.agent_app.logger.info(f"LLM selected resource: {uri}")
                    except Exception as e:
                        self.agent_app.logger.warning(f"Failed to read selected resource {uri}: {e}")
            
            # Step 4: Generate final response with selected resources
            response = await self.llm.generate_str(message=messages)
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

    
    async def chat_with_prompt(self, prompt_name: str, arguments: dict, additional_message: str = None) -> str:
        """
        Use a specific MCP prompt template with arguments.
        Useful for structured interactions.
        """
        try:
            # Create prompt with arguments (this formats the prompt and can include resources)
            prompt_messages = await self.agent.create_prompt(
                prompt_name=prompt_name,
                arguments=arguments,
                server_names=["user_management"],
            )
            
            # Combine with additional user message if provided
            messages = prompt_messages if not additional_message else [additional_message, *prompt_messages]
            
            response = await self.llm.generate_str(messages)
            return response
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    async def reset_conversation(self):
        """Reset conversation history"""
        if self.llm:
            self.llm.memory.clear()
            print("🔄 Conversation history cleared")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.agent:
            await self.agent.__aexit__(None, None, None)


async def main():
    """Main interactive chat loop"""
    print("\n" + "="*60)
    print("🤖 MCP AGENT CHAT CLIENT")
    print("="*60)
    print("Type your queries naturally. Examples:")
    print("  - 'Show me all users'")
    print("  - 'Create a new user named John with email john@example.com'")
    print("  - 'Get details for user with ID 1'")
    print("  - 'Create an admin user'")
    print("\nCommands:")
    print("  - 'exit' - quit the program")
    print("  - 'reset' - clear conversation history")
    print("  - 'help' - show available tools")
    print("="*60 + "\n")
    
    chat_agent = MCPChatAgent()
    
    async with app.run() as mcp_agent_app:
        try:
            await chat_agent.initialize(mcp_agent_app)
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            print("Make sure your MCP server is configured in mcp_agent.config.yaml")
            return
        
        # Chat loop
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() == "exit":
                        print("\n👋 Goodbye!")
                        break
                        
                    if user_input.lower() == "reset":
                        await chat_agent.reset_conversation()
                        continue
                    
                    if user_input.lower() == "help":
                        print("\n📋 Available capabilities:")
                        tools_result = await chat_agent.agent.list_tools()
                        for tool in tools_result.tools:
                            print(f"  - {tool.name}: {tool.description or 'No description'}")
                        print()
                        continue
                    
                    print()  # Blank line for readability
                    
                    # Choose your approach:
                    
                    # OPTION 1: Basic - only tools, no resources (DEFAULT)
                    # response = await chat_agent.chat(user_input)
                    
                    # OPTION 2: Include ALL resources automatically
                    # response = await chat_agent.chat(user_input, auto_include_resources=True)
                    
                    # OPTION 3: Let LLM decide which resources to use (2 LLM calls)
                    response = await chat_agent.chat_with_llm_resource_selection(user_input)

                    
                    print(f"\n🤖 Assistant: {response}\n")
                    print("-" * 60)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
        finally:
            # Clean up
            await chat_agent.cleanup()


async def batch_queries():
    """Process multiple queries in batch"""
    chat_agent = MCPChatAgent()
    
    async with app.run() as mcp_agent_app:
        try:
            await chat_agent.initialize(mcp_agent_app)
            
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
                
                response = await chat_agent.chat(query)
                print(f"\n🤖 Response: {response}\n")
                
                await asyncio.sleep(1)  # Rate limiting
        finally:
            await chat_agent.cleanup()


if __name__ == "__main__":
    # Interactive mode
    asyncio.run(main())
    
    # Or batch mode:
    # asyncio.run(batch_queries())
