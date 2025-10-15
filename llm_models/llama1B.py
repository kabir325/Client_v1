#!/usr/bin/env python3
"""
Dhenu2 Llama3.2-1B-Instruct Model Server
Lightweight LLM for basic agricultural queries
"""

import os
import torch
import gradio as gr
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama1BServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "KissanAI/Dhenu2-In-Llama3.2-1B-Instruct"
        self.load_model()
    
    def load_model(self):
        """Load the 1B model"""
        try:
            logger.info(f"🚀 Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations for smaller hardware
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        """Generate text response"""
        try:
            # Add system prompt for agricultural context
            system_prompt = "You are Dhenu2, an AI assistant specialized in Indian agriculture and farming. Provide helpful, practical advice."
            full_prompt = f"{system_prompt}\\n\\nUser: {prompt}\\nAssistant:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def create_gradio_interface(self):
        """Create Gradio web interface"""
        def chat_interface(prompt, max_tokens, temperature, top_p):
            if not prompt.strip():
                return "Please enter a prompt."
            
            logger.info(f"🔄 Processing: {prompt[:50]}...")
            response = self.generate_text(
                prompt=prompt,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            logger.info(f"✅ Response generated ({len(response)} chars)")
            return response
        
        # Create interface
        interface = gr.Interface(
            fn=chat_interface,
            inputs=[
                gr.Textbox(
                    lines=4, 
                    placeholder="Ask about agriculture, farming, crops, livestock...",
                    label="Your Question"
                ),
                gr.Slider(50, 512, value=256, label="Max Tokens"),
                gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
                gr.Slider(0.1, 1.0, value=0.9, label="Top P")
            ],
            outputs=gr.Textbox(
                lines=8,
                label="Dhenu2 Response"
            ),
            title="🌾 Dhenu2 Llama3.2-1B Agricultural AI",
            description="Lightweight AI assistant for Indian agriculture and farming queries. Optimized for basic hardware.",
            examples=[
                ["What are the best practices for rice cultivation during monsoon?"],
                ["How to prevent pest attacks in wheat crops?"],
                ["What is the ideal soil pH for tomato farming?"],
                ["When is the best time to plant cotton in Maharashtra?"]
            ]
        )
        
        return interface

def main():
    """Main function to start the server"""
    logger.info("🌾 Starting Dhenu2 Llama3.2-1B Server...")
    
    try:
        # Initialize server
        server = Llama1BServer()
        
        # Create and launch interface
        interface = server.create_gradio_interface()
        
        # Get port from environment or use default
        port = int(os.getenv('GRADIO_PORT', 7861))
        
        logger.info(f"🚀 Launching Gradio interface on port {port}")
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise

if __name__ == "__main__":
    main()