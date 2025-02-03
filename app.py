import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class QueryRequest(BaseModel):
    query: str

# Function to interact with DeepSeek via Ollama
def get_deepseek_response(query: str):
    try:
        # Run Ollama command without --prompt flag
        process = subprocess.Popen(
            ["ollama", "run", "deepseek-coder"],  # Remove --prompt flag
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'  # Explicitly set encoding to 'utf-8'
        )

        # Send the query to the model as input via stdin
        stdout, stderr = process.communicate(input=query)
        
        # Log output and error streams for debugging
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

        if process.returncode == 0:
            return stdout.strip()  # Return the model's output
        else:
            print(f"Ollama error: {stderr}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {stderr.strip()}")
    
    except subprocess.CalledProcessError as e:
        # Log subprocess errors
        print(f"Ollama failed with error: {e.stderr}")
        traceback.print_exc()  # Print stack trace for debugging
        raise HTTPException(status_code=500, detail=f"Error running DeepSeek model: {str(e)}")
    except Exception as e:
        # Catch any other exceptions and log them
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()  # Print stack trace for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# FastAPI Endpoint
@app.post("/ai-research")
async def ai_research_agent(request: QueryRequest):
    response = get_deepseek_response(request.query)
    return {"query": request.query, "summary": response}

# Run FastAPI with Uvicorn for Hugging Face Spaces
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
