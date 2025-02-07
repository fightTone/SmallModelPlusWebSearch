from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uvicorn
from typing import Optional, Dict, Any, List
import json
from duckduckgo_search import DDGS
import html2text
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_TOKENS_PER_THREAD = 500  # Limit tokens for each search result processing

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    include_web_search: Optional[bool] = False
    num_search_results: Optional[int] = 3

class ModelSelection(BaseModel):
    model_name: str

class WebSearcher:
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.ddgs = DDGS()
        
    async def generate_search_queries(self, client, prompt: str, num_queries: int = 3) -> List[str]:
        """Use AI to generate optimized search queries"""
        try:
            search_prompt = f"""Generate {num_queries} different search queries to find information about: {prompt}
            Make each query specific and focused on different aspects. Format as a simple list with each query on a new line starting with a hyphen (-)."""
            
            response = await client.generate_simple_response({
                "prompt": search_prompt,
                "max_tokens": 200,
                "temperature": 0.7
            })
            
            # Extract queries from response
            queries = [
                line.strip().replace('- ', '') 
                for line in response.split('\n') 
                if line.strip() and line.strip().startswith('-')
            ]
            
            # If we didn't get enough queries, use variations of the original prompt
            if len(queries) < num_queries:
                queries.extend([prompt] * (num_queries - len(queries)))
                
            return queries[:num_queries]
            
        except Exception as e:
            print(f"Error generating search queries: {str(e)}")
            # Fallback to using the original prompt
            return [prompt] * num_queries
        
    async def process_search_result(self, client, result: Dict, context: str) -> str:
        """Process and summarize a single search result"""
        summary_prompt = f"""Summarize this search result in the context of: {context}
        
        Title: {result['title']}
        Content: {result['body']}
        
        Provide a concise summary focusing on relevant information."""
        
        response = await client.generate_simple_response({
            "prompt": summary_prompt,
            "max_tokens": MAX_TOKENS_PER_THREAD,
            "temperature": 0.5
        })
        
        return response
        
    async def search(self, client, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Perform enhanced web search with AI processing"""
        try:
            # Generate optimized search queries
            search_queries = await self.generate_search_queries(client, query, num_results)
            
            all_results = []
            search_summaries = []
            
            # Perform searches for each generated query
            for search_query in search_queries:
                results = list(self.ddgs.text(search_query, max_results=1))
                if results:
                    all_results.extend(results)
                    
            # Process each result in parallel using ThreadPoolExecutor
            async def process_results():
                tasks = []
                for result in all_results[:num_results]:
                    summary = await self.process_search_result(client, result, query)
                    search_summaries.append({
                        'title': result['title'],
                        'summary': summary
                    })
            
            await process_results()
            
            # Combine summaries into final result
            formatted_results = "Web search results:\n\n"
            for i, result in enumerate(search_summaries, 1):
                formatted_results += f"{i}. {result['title']}\n"
                formatted_results += f"Summary: {result['summary']}\n\n"
            
            # Generate final combined summary
            final_summary_prompt = f"""Based on these search results, provide a comprehensive answer:
            {formatted_results}
            
            Question: {query}
            
            Provide a well-organized response that synthesizes the information."""
            
            final_summary = await client.generate_simple_response({
                "prompt": final_summary_prompt,
                "max_tokens": 1000,
                "temperature": 0.7
            })
            
            return {
                "raw_results": formatted_results,
                "final_summary": final_summary
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Web search error: {str(e)}")

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.current_model = None
        self.web_searcher = WebSearcher()
        
    def set_model(self, model_name: str):
        self.current_model = model_name
        
    def get_current_model(self) -> Optional[str]:
        return self.current_model
        
    async def generate_simple_response(self, request: Dict[str, Any]) -> str:
        """Generate a simple response without streaming"""
        if not self.current_model:
            raise HTTPException(status_code=400, detail="No model selected")
            
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.current_model,
            "prompt": request["prompt"],
            "options": {
                "temperature": request.get("temperature", 0.7),
                "num_predict": request.get("max_tokens", 500)
            }
        }
        
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            # Process the streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            full_response += json_response['response']
                    except json.JSONDecodeError:
                        continue
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")
        
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of all downloaded models"""
        endpoint = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json().get('models', [])
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")
            
    async def generate_response(self, request: PromptRequest) -> Dict[str, Any]:
        if not self.current_model:
            raise HTTPException(status_code=400, detail="No model selected. Please select a model first.")
            
        # Handle web search if requested
        enhanced_prompt = request.prompt
        search_results = None
        
        if request.include_web_search:
            search_data = await self.web_searcher.search(
                self,
                request.prompt, 
                request.num_search_results
            )
            enhanced_prompt = f"""Context from web search:
{search_data['raw_results']}

Based on the above search results and context, please answer:
{request.prompt}"""
            search_results = search_data['raw_results']
            
        endpoint = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.current_model,
            "prompt": enhanced_prompt,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
            
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            # Process the streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'response' in json_response:
                            full_response += json_response['response']
                    except json.JSONDecodeError:
                        continue
            
            response_data = {
                "response": full_response,
                "includes_web_search": request.include_web_search
            }
            
            if search_results:
                response_data["web_search_results"] = search_results
                
            return response_data
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

# Initialize Ollama client
ollama_client = OllamaClient(OLLAMA_BASE_URL)

@app.get("/models")
async def list_models():
    """List all downloaded models"""
    try:
        models = await ollama_client.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/select")
async def select_model(model_selection: ModelSelection):
    """Select a model to use for generation"""
    try:
        models = await ollama_client.list_models()
        available_models = [model['name'] for model in models]
        
        if model_selection.model_name not in available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_selection.model_name} not found. Available models: {available_models}"
            )
        
        ollama_client.set_model(model_selection.model_name)
        return {
            "message": f"Successfully selected model: {model_selection.model_name}",
            "current_model": model_selection.model_name
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/current")
async def get_current_model():
    """Get the currently selected model"""
    current_model = ollama_client.get_current_model()
    if not current_model:
        return {"message": "No model currently selected"}
    return {"current_model": current_model}

@app.post("/generate")
async def generate(request: PromptRequest):
    """Generate a response using the selected Ollama model"""
    try:
        response = await ollama_client.generate_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)