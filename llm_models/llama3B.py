#!/usr/bin/env python3
"""
Dhenu2 Llama3.2-3B-Instruct Model Server
Medium-sized LLM for comprehensive agricultural analysis
"""

import os
import torch
import gradio as gr
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama3BServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.text_gen = None
        self.model_name = "KissanAI/Dhenu2-In-Llama3.2-3B-Instruct"
        self.load_model()
    
    def load_model(self):
        """Load the 3B model"""
        try:
            logger.info(f"🚀 Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline for easier generation
            self.text_gen = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            logger.info(f"✅ Model loaded successfully on {device}!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        """Generate text response"""
        try:
            # Enhanced system prompt for 3B model
            system_prompt = """You are Dhenu2, an advanced AI assistant specialized in Indian agriculture, farming, and livestock management. 
            You provide detailed, practical, and scientifically accurate advice for farmers. Consider regional variations, 
            seasonal factors, and sustainable farming practices in your responses."""
            
            full_prompt = f"{system_prompt}\\n\\nUser: {prompt}\\nAssistant:"
            
            # Generate using pipeline
            outputs = self.text_gen(
                full_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract response
            response = outputs[0]['generated_text']
            
            # Clean response - extract only assistant part
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
        
        # Create interface with more advanced options
        interface = gr.Interface(
            fn=chat_interface,
            inputs=[
                gr.Textbox(
                    lines=5, 
                    placeholder="Ask detailed questions about agriculture, crop management, livestock, soil health...",
                    label="Your Agricultural Query"
                ),
                gr.Slider(100, 1024, value=512, label="Max Tokens"),
                gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
                gr.Slider(0.1, 1.0, value=0.9, label="Top P")
            ],
            outputs=gr.Textbox(
                lines=12,
                label="Dhenu2 Detailed Response"
            ),
            title="🌾 Dhenu2 Llama3.2-3B Advanced Agricultural AI",
            description="Medium-scale AI assistant providing comprehensive analysis for Indian agriculture, farming, and livestock management.",
            examples=[
                ["Provide a detailed crop rotation plan for a 10-acre farm in Punjab with wheat and rice."],
                ["What are the integrated pest management strategies for cotton farming in Gujarat?"],
                ["Explain the nutritional requirements and feeding schedule for dairy cattle in different seasons."],
                ["How to implement precision agriculture techniques for sugarcane cultivation?"],
                ["What are the soil health indicators and how to improve soil fertility organically?"]
            ]
        )
        
        return interface

def main():
    """Main function to start the server"""
    logger.info("🌾 Starting Dhenu2 Llama3.2-3B Server...")
    
    try:
        # Initialize server
        server = Llama3BServer()
        
        # Create and launch interface
        interface = server.create_gradio_interface()
        
        # Get port from environment or use default
        port = int(os.getenv('GRADIO_PORT', 7862))
        
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