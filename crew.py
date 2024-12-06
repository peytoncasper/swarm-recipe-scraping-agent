from crewai import Agent, Task, Crew, Process
from litellm import completion
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List, Dict, Optional, Union, TypedDict
import json
import os
import re
import crewai


# Initialize Azure OpenAI configuration
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2023-05-15"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["SERPER_API_KEY"] = ""
os.environ["GOOGLE_CSE_ID"] = ""

# Custom LiteLLM implementation for CrewAI
class LiteLLMWrapper:
    def __init__(self, model_name="gpt-35-turbo"):
        self.model_name = model_name

    def __call__(self, messages, **kwargs):
        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message
        except Exception as e:
            print(f"Error calling LiteLLM: {str(e)}")
            raise

# Initialize LiteLLM wrapper and tools
llm = LiteLLMWrapper(model_name="azure/gpt-35-turbo")
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define the agents
recipe_finder = Agent(
    role='Recipe Finder',
    goal='Find the best recipes based on user requirements',
    backstory="""You are an expert at finding recipes online. You understand cooking techniques, 
    ingredients, and can identify authentic recipes from reliable sources.""",
    tools=[search_tool, scrape_tool],
    llm=llm,
)

recipe_analyzer = Agent(
    role='Recipe Analyzer',
    goal='Analyze recipes and extract structured information',
    backstory="""You are an expert at analyzing recipes and extracting key information like 
    ingredients, instructions, cooking times, and serving sizes. You ensure all information
    is properly structured and validated.""",
    tools=[scrape_tool],
    llm=llm,
)

recipe_formatter = Agent(
    role='Recipe Formatter',
    goal='Format recipe information into a standardized JSON structure',
    backstory="""You are a specialist in data structuring and formatting. You take recipe
    information and convert it into a clean, standardized JSON format that follows the
    specified schema exactly.""",
    llm=llm,
)

# Define the tasks
def create_recipe_tasks(recipe_query: str) -> List[Task]:
    return [
        Task(
            description="""
                Find a recipe for {recipe_query} from reliable sources.
                Use the search tool to find authentic recipes from reputable cooking websites.
                Evaluate multiple options and select the best match based on authenticity and user reviews.
                You must include the URL of the chosen recipe in your response.
            """,
            agent=recipe_finder,
            expected_output="""
                A detailed response containing:
                - The selected recipe URL
                - Why this recipe was chosen
                - Brief overview of the recipe
            """,
        ),
        Task(
            description=f"""
                Analyze the provided recipe and extract all relevant information.
                Use the URL from the previous task's response to scrape and analyze the recipe.
                Make sure to capture:
                - Complete list of ingredients with amounts
                - Detailed step-by-step instructions
                - Cooking times and temperatures
                - Serving size and yield
                - Any special notes or tips
                Include the source URL in your response for the next task.
            """,
            agent=recipe_analyzer,
            expected_output="""
                A comprehensive breakdown of the recipe including all ingredients,
                instructions, timings, and additional information, properly organized
                and ready for formatting. Must include the source URL.
            """,
        ),
        Task(
            description=f"""
                Format the recipe information into a standardized JSON structure.
                Make sure to include the source URL from the previous task's response.
                Follow this exact schema:
                {{
                    "title": string,
                    "description": string,
                    "prep_time": string,
                    "cook_time": string,
                    "servings": number,
                    "ingredients": [
                        {{
                            "item": string,
                            "amount": string,
                            "unit": string
                        }}
                    ],
                    "instructions": [
                        {{
                            "step_number": number,
                            "instruction": string
                        }}
                    ],
                    "source_url": string
                }}
            """,
            agent=recipe_formatter,
            expected_output="""
                A valid JSON object containing the complete recipe information,
                strictly following the provided schema, including the source URL.
            """,
            output_file="recipe.json",
        )
    ]

# Create the crew
def main() -> Dict:
    """Find, analyze, and format a recipe based on the user's query"""
    recipe_query = "spaghetti carbonara"
    crew = Crew(
        agents=[recipe_finder, recipe_analyzer, recipe_formatter],
        tasks=create_recipe_tasks(recipe_query),
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff()
    
    try:
        with open('recipe.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse recipe result", 
                "raw_result": result
            }
        
if __name__ == "__main__":
    main()
