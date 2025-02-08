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

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class PromptRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Message]] = []
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
        
    async def generate_search_queries(self, client, prompt: str, conversation_history=None, num_queries: int = 3) -> List[str]:
        """Use AI to generate optimized search queries based on context"""
        try:
            # Ensure conversation_history is a list
            conversation_history = conversation_history if isinstance(conversation_history, list) else []
            # print(conversation_history)
            
            # Build context for query generation
            context = ""
            if conversation_history and len(conversation_history) > 0:
                # Filter only user messages
                user_messages = [msg for msg in conversation_history if msg.role == "user"]
                if user_messages:
                    context = "Previous user questions:\n"
                    for msg in user_messages[-3:]:  # Last 3 user messages
                        context += f"- {msg.content}\n"
                    context += "\nCurrent question: " + prompt + "\n"
            else:
                context = "Question: " + prompt + "\n"

            search_prompt = f"""{context}
            Based on this {'previous questions' if conversation_history else 'question'}, generate {num_queries} specific web-search queries to find relevant information.
            {'Focus on maintaining context from the previous messages.' if conversation_history else 'Focus on the main aspects of the question.'}
            Make each query specific and targeted.
            Format as a simple list with each query on a new line starting with a hyphen (-)."""
            print(search_prompt)
            
            response = await client.generate_simple_response({
                "prompt": search_prompt,
                "max_tokens": 200,
                "temperature": 0.7
            })
            print(response)
            
            # Extract queries from response
            queries = [
                line.strip().replace('- ', '') 
                for line in response.split('\n') 
                if line.strip() and line.strip().startswith('-')
            ]
            
            # If we didn't get enough queries, use variations with context
            if len(queries) < num_queries:
                base_query = prompt
                if conversation_history and len(conversation_history) >= 2:
                    last_topic = conversation_history[-2].content
                    base_query = f"{last_topic} {prompt}"
                queries.extend([base_query] * (num_queries - len(queries)))
                
            return queries[:num_queries]
            
        except Exception as e:
            print(f"Error generating search queries: {str(e)}")
            return [prompt] * num_queries  # Simple fallback
        
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
        
    async def search(self, client, query: str, conversation_history: str, num_results: int = 3) -> Dict[str, Any]:
        """Perform enhanced web search with AI processing"""
        try:
            # Generate optimized search queries
            search_queries = await self.generate_search_queries(client, query, conversation_history, num_results)
            
            all_results = []
            search_summaries = []
            
            # Perform searches for each generated query
            for search_query in search_queries:
                results = list(self.ddgs.text(search_query, max_results=1))
                if results:
                    all_results.extend(results)
                    
            # Process each result to extract key points
            async def process_results():
                tasks = []
                for i, result in enumerate(all_results[:num_results], 1):
                    points_prompt = f"""Extract key factual points from this source:
                    Title: {result['title']}
                    Content: {result['body']}
                    
                    List 3-5 main factual points, focusing on information relevant to: {query}
                    Format as bullet points."""
                    
                    points = await client.generate_simple_response({
                        "prompt": points_prompt,
                        "max_tokens": MAX_TOKENS_PER_THREAD,
                        "temperature": 0.5
                    })
                    
                    search_summaries.append({
                        'title': result['title'],
                        'source_num': i,
                        'key_points': points
                    })
            
            await process_results()
            
            # Format results with source numbers
            formatted_results = "Web Search Results:\n\n"
            for result in search_summaries:
                formatted_results += f"Source {result['source_num']}: {result['title']}\n"
                formatted_results += f"Key Points:\n{result['key_points']}\n\n"
            
            # Generate final combined summary
            final_summary_prompt = f"""Extract key points from these search results:
            {formatted_results}
            
            Question: {query}
            
            List the most important factual points from the sources using bullet points.
            For each point, indicate which source it came from (Source 1, 2, or 3).
            Organize the points by relevance to the question.
            Only include factual information that appears in the sources."""
            
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
        # Build context from conversation history
        context = ""
        if request.conversation_history:
            context = "Previous conversation:\n"
            for msg in request.conversation_history[-3:]:  # Last 3 messages for context
                context += f"{msg.role.title()}: {msg.content}\n"
            context += "\nCurrent question: "
            
        enhanced_prompt = f"{context}{request.prompt}"
        search_results = None
        
        if request.include_web_search:
            search_data = await self.web_searcher.search(
                self,
                request.prompt, 
                request.conversation_history,
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