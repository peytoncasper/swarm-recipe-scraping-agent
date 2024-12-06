![Recipe Swarm Agent Banner](assets/banner.png)

# Recipe Swarm Agent

A multi-agent system that finds and formats recipes using Azure OpenAI and web scraping capabilities.

## Overview

This project demonstrates the power of AI agents for automated recipe extraction using OpenAI Swarm, a minimalist framework focused on simplicity and customization. The system efficiently searches, scrapes, and formats recipes into structured JSON data.

[Read the full blog post here](URL_TO_BE_ADDED)

## Features

- Web search integration using Google Custom Search API
- Recipe scraping and formatting
- Multi-agent orchestration:
  - Recipe Finder Agent: Searches and recommends recipes
  - Format Agent: Structures recipes into standardized JSON format
  - Orchestrator Agent: Coordinates between specialized agents

## Key Components

- **Google Search Tool**: Utilizes Google's Custom Search API to find recipes
- **Scraping Tool**: Uses BeautifulSoup for efficient webpage text extraction
- **JSON Formatter**: Leverages GPT-4 to transform raw recipe text into structured JSON

## Output Format

Recipes are formatted into a standardized JSON structure:

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/recipe-swarm-agent.git
   cd recipe-swarm-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your:
   - Azure OpenAI API key and endpoint
   - Google Custom Search API key
   - Google Custom Search Engine ID

4. Run the application:
   ```bash
   python swarm.py
   ```