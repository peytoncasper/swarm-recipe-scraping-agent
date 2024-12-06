import os
from openai import AzureOpenAI
from typing import List, Dict, Optional, Union, TypedDict
import json
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import re
# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-03-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class RecipeIngredient(TypedDict):
    item: str
    amount: Optional[str]
    unit: Optional[str]

class RecipeStep(TypedDict):
    step_number: int
    instruction: str

class RecipeSchema(TypedDict):
    title: str
    description: Optional[str]
    prep_time: Optional[str]
    cook_time: Optional[str]
    servings: Optional[int]
    ingredients: List[RecipeIngredient]
    instructions: List[RecipeStep]
    source_url: Optional[str]

def search_web(query: str) -> str:
    """Perform web search using Google Custom Search API"""
    try:
        print(f"\n=== Search Web Tool ===")
        print(f"Query: {query}")
        
        # Initialize the Custom Search API service
        service = build(
            "customsearch", "v1",
            developerKey=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create a Custom Search Engine ID
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        # Perform the search
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=5  # Number of results to return
        ).execute()
        
        # Format the results
        formatted_results = []
        if 'items' in result:
            for item in result['items']:
                formatted_results.append({
                    'title': item['title'],
                    'link': item['link'],
                    'snippet': item['snippet']
                })
        
        print("Search Results:")
        print(json.dumps(formatted_results, indent=2))
        return json.dumps(formatted_results, indent=2)
        
    except Exception as e:
        error_msg = f"Error performing search: {str(e)}"
        print(error_msg)
        return error_msg

def extract_recipe_json(text: str, url: str = None) -> Dict:
    """Extract recipe information from text and format as JSON"""
    try:
        messages = [
            {"role": "system", "content": """You are a recipe extraction specialist. Extract recipe details from the input text and format them according to this schema:
            {
                "ingredients": [
                    "string" // Each ingredient with amount and unit
                ],
                "instructions": [
                    "string" // Each step as a clear instruction
                ],
                "source_url": "string" // URL where the recipe was found
            }
            
            Rules:
            1. Always respond with valid JSON that matches this schema exactly
            2. Extract ingredients with their measurements
            3. Extract instructions as clear, sequential steps
            4. Clean up and standardize the formatting
            5. Include the source URL in the response
            6. Format your entire response as a single JSON object"""},
            {"role": "user", "content": text}
        ]
        
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            # Parse the response to ensure it's valid JSON
            recipe_json = json.loads(response.choices[0].message.content)
            # Add the source URL if provided
            if url:
                recipe_json["source_url"] = url
            return recipe_json
        except json.JSONDecodeError:
            return {
                "ingredients": [],
                "instructions": [],
                "source_url": url if url else None,
                "error": "Failed to parse LLM response as JSON"
            }
            
    except Exception as e:
        return {
            "error": f"Failed to extract recipe: {str(e)}",
            "source_url": url if url else None
        }

def scrape_webpage(url: str, format_as_json: bool = False) -> Union[str, Dict]:
    """Enhanced web scraping with optional JSON formatting for recipes"""
    try:
        print(f"\n=== Scrape Webpage Tool ===")
        print(f"URL: {url}")
        print(f"Format as JSON: {format_as_json}")
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        if format_as_json:
            result = extract_recipe_json(text, url)
            print("\nExtracted Recipe JSON:")
            print(json.dumps(result, indent=2))
            return result
            
        print("\nScraped Text Preview (first 500 chars):")
        print(text[:500] + "...")
        return text
        
    except Exception as e:
        error_msg = f"Error scraping webpage: {str(e)}"
        print(error_msg)
        return error_msg

class SwarmAgent:
    def __init__(self, name: str, description: str, tools: List[Dict] = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.tool_implementations = {
            "search_web": search_web,
            "scrape_webpage": scrape_webpage
        }
        
        # For Azure OpenAI, we don't create an assistant - we'll use direct chat completion
        self.system_message = f"""You are {name}. {description}
        You have access to tools for searching the web and scraping webpages.
        Use these tools when needed to provide accurate and up-to-date information."""

    def run(self, prompt: str) -> Dict:
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=messages,
                functions=self.tools if self.tools else None,
                temperature=0.7
            )
            
            # Handle function calling if present
            while response.choices[0].finish_reason == "function_call":
                function_call = response.choices[0].message.function_call
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)
                
                print(f"\n=== Tool Call ===")
                print(f"Function: {function_name}")
                print(f"Arguments: {json.dumps(function_args, indent=2)}")
                
                # Execute the function
                if function_name in self.tool_implementations:
                    function_response = self.tool_implementations[function_name](**function_args)
                else:
                    function_response = f"Error: Function {function_name} not found"
                
                # Add the function call and result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_name,
                        "arguments": function_call.arguments
                    }
                })
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response)
                })
                
                # Get a new response from the model
                response = client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=messages,
                    functions=self.tools if self.tools else None,
                    temperature=0.7
                )
            
            return {
                "data": {
                    "output": response.choices[0].message.content
                }
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                "data": {
                    "output": error_msg
                }
            }

# Built-in tool definitions
SEARCH_TOOL = {
    "name": "search_web",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

SCRAPE_TOOL = {
    "name": "scrape_webpage",
    "description": "Scrape content from a webpage with optional JSON formatting for recipes",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "format_as_json": {
                "type": "boolean",
                "description": "Whether to format the output as a recipe JSON"
            }
        },
        "required": ["url"]
    }
}

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "prep_time": {"type": "string"},
        "cook_time": {"type": "string"},
        "servings": {"type": "integer"},
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "amount": {"type": "string"},
                    "unit": {"type": "string"}
                }
            }
        },
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "instruction": {"type": "string"}
                }
            }
        },
        "source_url": {"type": "string"}
    }
}

class OrchestratorAgent(SwarmAgent):
    def __init__(self, sub_agents: Dict[str, SwarmAgent]):
        super().__init__(
            name="Orchestrator",
            description="I coordinate between multiple specialized agents to solve complex tasks",
            tools=[SEARCH_TOOL, SCRAPE_TOOL]
        )
        self.sub_agents = sub_agents

    def run(self, prompt: str) -> Dict:
        try:
            # First, determine which agent(s) should handle the task
            planning_messages = [
                {"role": "system", "content": f"""You are an orchestrator that delegates tasks to specialized agents.
                Available agents: {', '.join(self.sub_agents.keys())}. 
                Respond with a JSON array of tasks, where each task has:
                - agent_name: which agent to use
                - sub_task: what specific task they should perform
                Example: [
                    {{"agent_name": "recipe_finder", "sub_task": "find a pasta recipe"}},
                    {{"agent_name": "formatter", "sub_task": "format the recipe"}}
                ]"""},
                {"role": "user", "content": prompt}
            ]
            
            planning_response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=planning_messages,
                temperature=0.7
            )
            
            print("\n=== Orchestrator Planning ===")
            print(planning_response.choices[0].message.content)
            
            # Parse the planning response to get agent assignments
            try:
                agent_tasks = json.loads(planning_response.choices[0].message.content)
                if not isinstance(agent_tasks, list):
                    agent_tasks = [agent_tasks]  # Convert single task to list
            except json.JSONDecodeError:
                # Fallback to using the first available agent if parsing fails
                agent_tasks = [{"agent_name": list(self.sub_agents.keys())[0], "sub_task": prompt}]
            
            # Execute tasks with appropriate agents and collect results
            results = []
            recipe_url = None  # Track the URL found by recipe finder
            
            for task in agent_tasks:
                agent_name = task.get("agent_name")
                sub_task = task.get("sub_task")
                
                print(f"\n=== {agent_name} Task ===")
                print(f"Task: {sub_task}")
                
                if agent_name and agent_name in self.sub_agents:
                    agent_response = self.sub_agents[agent_name].run(sub_task)
                    print(f"\n{agent_name} Response:")
                    print(json.dumps(agent_response["data"]["output"], indent=2))
                    
                    # If this is the recipe finder, try to extract the URL
                    if agent_name == "recipe_finder":
                        try:
                            # Look for a URL in the response
                            response_data = agent_response["data"]["output"]
                            if isinstance(response_data, str):
                                urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', response_data)
                                if urls:
                                    recipe_url = urls[0]
                                    print(f"\nFound recipe URL: {recipe_url}")
                    
                        except Exception as e:
                            print(f"Error extracting URL: {str(e)}")
                    
                    results.append({
                        "agent": agent_name,
                        "task": sub_task,
                        "response": agent_response["data"]["output"],
                        "url": recipe_url if agent_name == "recipe_finder" else None
                    })
            
            # Pass the URL to the formatter if available
            if recipe_url:
                for result in results:
                    if result["agent"] == "formatter":
                        try:
                            formatted_response = json.loads(result["response"])
                            formatted_response["source_url"] = recipe_url
                            result["response"] = json.dumps(formatted_response)
                        except:
                            pass
            
            # Synthesize results
            synthesis_messages = [
                {"role": "system", "content": "Synthesize the results from multiple agents into a coherent response."},
                {"role": "user", "content": f"Results: {json.dumps(results, indent=2)}"}
            ]
            
            synthesis_response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=synthesis_messages,
                temperature=0.7
            )
            
            print("\n=== Final Synthesis ===")
            print(synthesis_response.choices[0].message.content)
            
            return {
                "data": {
                    "output": synthesis_response.choices[0].message.content,
                    "agent_results": results
                }
            }
            
        except Exception as e:
            error_msg = f"Orchestration Error: {str(e)}"
            print(f"\n=== Error ===\n{error_msg}")
            return {
                "data": {
                    "output": error_msg,
                    "error_details": str(e)
                }
            }
        
# Add this function near the top of the file with other functions
def save_recipe_to_file(recipe_json: str, filename: str = "recipe.json") -> None:
    """Save the formatted recipe to a JSON file"""
    try:
        # Parse the string to ensure it's valid JSON before saving
        recipe_data = json.loads(recipe_json)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recipe_data, f, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {str(e)}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


# Add example specialized agents at the bottom of the file
if __name__ == "__main__":
    # Create specialized agents
    recipe_finder = SwarmAgent(
        name="Recipe Finder",
        description=(
            "This agent specializes in finding and recommending recipes based on cuisine preferences. "
            "It searches through recipe databases and websites to find authentic recipes matching the requested cuisine type. "
            "The agent provides detailed ingredient lists, cooking instructions, and helpful tips for preparation."
        ),
        tools=[SEARCH_TOOL, SCRAPE_TOOL]
    )
    
    formatter_agent = SwarmAgent(
        name="Format Agent",
        description=(
            "This agent specializes in parsing recipe text into a structured JSON format. "
            "It extracts key recipe components like ingredients, instructions, and timing "
            "and formats them according to a standardized schema."
        ),
        tools=[SCRAPE_TOOL]  # Only needs scraping tool
    )
    
    # Modify the run method of the formatter agent to enforce JSON schema
    def formatted_run(self, prompt: str) -> Dict:
        messages = [
            {"role": "system", "content": f"""You are a recipe formatting specialist. 
            Extract recipe details from the input and format them according to this exact schema:
            {json.dumps(FORMAT_SCHEMA, indent=2)}
            
            Rules:
            1. Always respond with valid JSON that matches this schema exactly
            2. If any fields are missing, use null or empty lists as appropriate
            3. Ensure all ingredients are properly separated into item, amount, and unit
            4. All string fields must be strings, integers must be numbers
            5. Format your entire response as a single JSON object"""},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=messages,
            temperature=0.7
        )
        
        try:
            # Parse the response to ensure it's valid JSON
            formatted_json = json.loads(response.choices[0].message.content)
            return {
                "data": {
                    "output": json.dumps(formatted_json, indent=2)
                }
            }
        except json.JSONDecodeError:
            return {
                "data": {
                    "output": "Error: Failed to generate valid JSON format",
                    "raw_response": response.choices[0].message.content
                }
            }
    
    # Patch the formatter agent's run method
    formatter_agent.run = formatted_run.__get__(formatter_agent)
    
    # Create orchestrator with sub-agents
    orchestrator = OrchestratorAgent({
        "recipe_finder": recipe_finder,
        "formatter": formatter_agent
    })
    
    # Test the orchestrator with formatting and save results
    response = orchestrator.run(
        "Find me a good spaghetti carbonara recipe and format it properly"
    )
    
    # Extract the formatted recipe from the formatter agent's response
    try:
        agent_results = response['data']['agent_results']
        formatter_result = next(
            result['response'] for result in agent_results 
            if result['agent'] == 'formatter'
        )
        
        # Save the formatted recipe to file
        save_recipe_to_file(formatter_result)
        print("Recipe saved to recipe.json")
        
        # Also print the full response
        print(json.dumps(response['data'], indent=2))
    except Exception as e:
        print(f"Error processing response: {str(e)}")

