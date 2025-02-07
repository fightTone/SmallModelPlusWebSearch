from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uvicorn
from typing import Optional, Dict, Any, List
import json
from duckduckgo_search import DDGS
import html2text
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
        
    async def search(self, query: str, num_results: int = 3) -> str:
        """Perform web search and return formatted results"""
        try:
            results = list(self.ddgs.text(query, max_results=num_results))
            
            if not results:
                return "No search results found."
                
            formatted_results = "Web search results:\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. Title: {result['title']}\n"
                formatted_results += f"Summary: {result['body']}\n\n"
                
            return formatted_results
            
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
            
        # Enrich prompt with web search if requested
        enhanced_prompt = request.prompt
        if request.include_web_search:
            search_results = await self.web_searcher.search(
                request.prompt, 
                request.num_search_results
            )
            enhanced_prompt = f"""Context from web search:
{search_results}

Based on the above context, please answer:
{request.prompt}"""
            
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
                    except json.JSONDecodeError as e:
                        continue
            
            response_data = {
                "response": full_response,
                "includes_web_search": request.include_web_search,
            }
            
            # Include web search results if they were used
            if request.include_web_search:
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