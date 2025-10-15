#!/usr/bin/env python3
"""
Dhenu2 Llama3.1-8B-Instruct Model Server
Large-scale LLM for advanced agricultural research and analysis
"""

import os
import torch
import gradio as gr
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import bitsandbytes as bnb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama8BServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.text_gen = None
        self.model_name = "KissanAI/Dhenu2-In-Llama3.1-8B-Instruct"
        self.load_model()
    
    def load_model(self):
        """Load the 8B model with quantization"""
        try:
            logger.info(f"🚀 Loading {self.model_name} with 4-bit quantization...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure 4-bit quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.text_gen = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer
            )
            
            logger.info("✅ 8B Model loaded successfully with quantization!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        """Generate text response"""
        try:
            # Comprehensive system prompt for 8B model
            system_prompt = """You are Dhenu2, an expert AI agricultural consultant with deep knowledge of Indian farming systems, 
            crop science, livestock management, agricultural economics, and sustainable farming practices. You provide comprehensive, 
            research-backed advice considering regional climate, soil conditions, market trends, and government policies. 
            Your responses are detailed, practical, and scientifically accurate."""
            
            full_prompt = f"{system_prompt}\\n\\nUser: {prompt}\\nAssistant:"
            
            # Generate using pipeline with advanced parameters
            response = self.text_gen(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Clean response - extract only assistant part
            if "Assistant:" in generated_text:
                generated_text = generated_text.split("Assistant:")[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def create_gradio_interface(self):
        """Create advanced Gradio web interface"""
        def chat_interface(prompt, max_tokens, temperature, top_p, repetition_penalty):
            if not prompt.strip():
                return "Please enter a prompt."
            
            logger.info(f"🔄 Processing complex query: {prompt[:50]}...")
            response = self.generate_text(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            logger.info(f"✅ Comprehensive response generated ({len(response)} chars)")
            return response
        
        # Create advanced interface
        interface = gr.Interface(
            fn=chat_interface,
            inputs=[
                gr.Textbox(
                    lines=6, 
                    placeholder="Ask complex questions about agricultural research, policy analysis, market trends, advanced farming techniques...",
                    label="Your Advanced Agricultural Query"
                ),
                gr.Slider(200, 2048, value=512, label="Max New Tokens"),
                gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
                gr.Slider(0.1, 1.0, value=0.9, label="Top P"),
                gr.Slider(1.0, 1.5, value=1.1, label="Repetition Penalty")
            ],
            outputs=gr.Textbox(
                lines=15,
                label="Dhenu2 Expert Analysis"
            ),
            title="🌾 Dhenu2 Llama3.1-8B Expert Agricultural AI",
            description="Large-scale AI expert providing comprehensive research-level analysis for Indian agriculture, policy, and advanced farming systems.",
            examples=[
                ["Analyze the impact of climate change on wheat production in North India and suggest adaptation strategies."],
                ["Develop a comprehensive business plan for setting up a 50-acre organic farm with integrated livestock."],
                ["Compare different irrigation technologies for water-scarce regions and their economic viability."],
                ["Explain the role of biotechnology in improving crop yields while maintaining sustainability."],
                ["Provide a detailed analysis of government agricultural policies and their impact on small farmers."],
                ["Design an integrated pest and disease management system for a multi-crop farming system."]
            ]
        )
        
        return interface

def main():
    """Main function to start the server"""
    logger.info("🌾 Starting Dhenu2 Llama3.1-8B Expert Server...")
    
    try:
        # Initialize server
        server = Llama8BServer()
        
        # Create and launch interface
        interface = server.create_gradio_interface()
        
        # Get port from environment or use default
        port = int(os.getenv('GRADIO_PORT', 7863))
        
        logger.info(f"🚀 Launching Expert Gradio interface on port {port}")
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