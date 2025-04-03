from langchain.llms import LlamaCpp
from typing import Optional
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

def setup_llm(model_path: str, 
              n_ctx: int = 4048,
              n_threads: Optional[int] = None,
              temperature: float = 0.7,
              n_gpu_layers: Optional[int] = None) -> LlamaCpp:
    """Configure and return a LlamaCpp LLM instance.
    
    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size
        n_threads: Number of CPU threads to use (None for auto)
        temperature: Model temperature
        n_gpu_layers: Number of layers to offload to GPU (None for auto)
        
    Returns:
        Configured LlamaCpp instance
    """
    gpu_params = {}
    if HAS_GPU:
        gpu_params = {
            'n_gpu_layers': n_gpu_layers or -1,  # -1 = offload all layers
            'main_gpu': 0,  # Use first GPU
            'tensor_split': None  # Evenly split across GPUs
        }
        print("GPU acceleration enabled")
    
    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=temperature,
        verbose=True,
        **gpu_params
    )

def test_llm(llm: LlamaCpp):
    """Test the LLM with a simple prompt."""
    prompt = "What is the capital of France?"
    print(f"\nTesting LLM with prompt: '{prompt}'")
    response = llm(prompt)
    print("LLM Response:", response)

if __name__ == "__main__":
    # Example usage with downloaded model
    print("LLM Configuration Module")
    llm = setup_llm("models/llama-2-7b-chat.Q4_K_M.gguf")
    test_llm(llm)
