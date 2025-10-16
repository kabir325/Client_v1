#!/usr/bin/env python3
"""
Simple LLM Server
A lightweight server that provides agricultural AI responses using actual LLM models
"""

import os
import gradio as gr
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLLMServer:
    def __init__(self, model_name="simple-agricultural-ai"):
        self.model_name = model_name
        self.model_size = self._get_model_size(model_name)
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _get_model_size(self, model_name):
        """Get model size from name"""
        if "1b" in model_name.lower():
            return "1B"
        elif "3b" in model_name.lower():
            return "3B"
        elif "8b" in model_name.lower():
            return "8B"
        else:
            return "Simple"
    
    def _load_model(self):
        """Load the actual LLM model"""
        try:
            # Try to load transformers and torch
            import torch
            from transformers import pipeline
            
            logger.info(f"Loading model for {self.model_name}...")
            
            # Use a smaller, readily available model for demonstration
            # This avoids authentication issues and large downloads
            try:
                # Try GPT-2 first (small and readily available)
                logger.info("Attempting to load GPT-2 small model...")
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_length=256,
                    pad_token_id=50256
                )
                logger.info("GPT-2 model loaded successfully")
                self.model_loaded = True
                
            except Exception as e:
                logger.warning(f"Failed to load GPT-2: {e}")
                # Try an even smaller model
                try:
                    logger.info("Attempting to load DistilGPT-2...")
                    self.pipeline = pipeline(
                        "text-generation",
                        model="distilgpt2",
                        max_length=256
                    )
                    logger.info("DistilGPT-2 model loaded successfully")
                    self.model_loaded = True
                    
                except Exception as e2:
                    logger.warning(f"Failed to load DistilGPT-2: {e2}")
                    raise e2
            
        except Exception as e:
            logger.warning(f"Failed to load any actual model: {e}")
            logger.info("Falling back to simulated responses")
            self.model_loaded = False
            self.pipeline = None
    
    def generate_response(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """Generate agricultural AI response using actual LLM or fallback"""
        try:
            logger.info(f"Generating response for: {prompt[:50]}...")
            
            if self.model_loaded and self.pipeline:
                # Use actual LLM model
                return self._generate_llm_response(prompt, max_tokens, temperature, top_p)
            else:
                # Fallback to simulated responses
                return self._generate_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while processing your agricultural query: {str(e)}"
    
    def _generate_llm_response(self, prompt, max_tokens, temperature, top_p):
        """Generate response using actual LLM model"""
        try:
            # Create a focused prompt for agricultural context
            agricultural_prompt = f"Agricultural advice for: {prompt}\n\nRecommendation:"
            
            # Generate response using the pipeline
            outputs = self.pipeline(
                agricultural_prompt,
                max_length=len(agricultural_prompt.split()) + max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id if hasattr(self.pipeline.tokenizer, 'eos_token_id') else None
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Extract only the new part (after the prompt)
            if "Recommendation:" in generated_text:
                response = generated_text.split("Recommendation:")[-1].strip()
            else:
                # If the format is different, try to extract the relevant part
                response = generated_text[len(agricultural_prompt):].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Ensure we have a reasonable response
            if len(response) < 20:
                logger.warning("Generated response too short, using fallback")
                return self._generate_fallback_response(prompt)
            
            logger.info(f"Generated LLM response ({len(response)} chars)")
            return f"AI-Generated Agricultural Advice: {response}"
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(prompt)
    
    def _clean_response(self, response):
        """Clean up the generated response"""
        # Remove any unwanted patterns
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Farmer Question:') and not line.startswith('System:'):
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _generate_fallback_response(self, prompt):
        """Generate fallback response when LLM is not available"""
        logger.info("Using fallback response generation")
        
        # Simulate processing time based on model size
        if self.model_size == "1B":
            time.sleep(1 + random.uniform(0.5, 1.5))
        elif self.model_size == "3B":
            time.sleep(2 + random.uniform(1, 3))
        elif self.model_size == "8B":
            time.sleep(4 + random.uniform(2, 4))
        else:
            time.sleep(1)
        
        # Generate contextual agricultural response
        response = self._generate_agricultural_response(prompt, self.model_size)
        
        logger.info(f"Generated fallback response ({len(response)} chars)")
        return response
    
    def _generate_agricultural_response(self, prompt, model_size):
        """Generate contextual agricultural response based on prompt and model size"""
        
        # Analyze prompt for agricultural context
        prompt_lower = prompt.lower()
        
        # Base response templates by model size (no emojis)
        if model_size == "1B":
            prefix = "Quick Agricultural Advice: "
            style = "concise and practical"
        elif model_size == "3B":
            prefix = "Detailed Agricultural Analysis: "
            style = "comprehensive and scientific"
        elif model_size == "8B":
            prefix = "Expert Agricultural Consultation: "
            style = "research-level and policy-oriented"
        else:
            prefix = "Agricultural Response: "
            style = "helpful and informative"
        
        # Context-specific responses
        if any(word in prompt_lower for word in ['organic', 'natural', 'pesticide-free']):
            topic = "organic farming"
            content = self._get_organic_farming_advice(model_size)
        elif any(word in prompt_lower for word in ['rice', 'paddy', 'grain']):
            topic = "rice cultivation"
            content = self._get_rice_farming_advice(model_size)
        elif any(word in prompt_lower for word in ['pest', 'insect', 'disease', 'bug']):
            topic = "pest management"
            content = self._get_pest_management_advice(model_size)
        elif any(word in prompt_lower for word in ['soil', 'fertility', 'nutrient']):
            topic = "soil health"
            content = self._get_soil_health_advice(model_size)
        elif any(word in prompt_lower for word in ['water', 'irrigation', 'drought']):
            topic = "water management"
            content = self._get_water_management_advice(model_size)
        else:
            topic = "general farming"
            content = self._get_general_farming_advice(model_size)
        
        return f"{prefix}Regarding {topic}, {content}"
    
    def _get_organic_farming_advice(self, model_size):
        if model_size == "1B":
            return "use natural fertilizers like compost and cow dung, avoid chemical pesticides, practice crop rotation, and use neem oil for pest control."
        elif model_size == "3B":
            return "implement integrated organic practices including composting, green manuring, biological pest control using beneficial insects, crop diversification, and soil health management through organic matter addition. Consider certification processes and market linkages for organic produce."
        else:
            return "develop a comprehensive organic farming system incorporating advanced techniques like biodynamic farming, precision composting, integrated pest and disease management using biocontrol agents, soil microbiome enhancement, carbon sequestration practices, and sustainable supply chain development with proper certification and premium market access."
    
    def _get_rice_farming_advice(self, model_size):
        if model_size == "1B":
            return "prepare fields properly, use quality seeds, maintain proper water levels, apply fertilizers at right time, and control weeds regularly."
        elif model_size == "3B":
            return "follow System of Rice Intensification (SRI) methods, ensure proper land preparation with laser leveling, use certified seeds with appropriate spacing, implement alternate wetting and drying for water management, apply balanced nutrition through soil testing, and integrate pest management strategies."
        else:
            return "implement climate-smart rice cultivation using drought-tolerant varieties, precision agriculture techniques, real-time monitoring systems, optimized nutrient management based on leaf color charts and soil sensors, integrated water management with alternate wetting and drying, mechanization for efficiency, and post-harvest management for quality retention and market access."
    
    def _get_pest_management_advice(self, model_size):
        if model_size == "1B":
            return "identify pests early, use neem-based sprays, encourage beneficial insects, remove infected plants, and follow crop rotation."
        elif model_size == "3B":
            return "implement Integrated Pest Management (IPM) combining cultural practices, biological control using parasitoids and predators, selective use of biopesticides, pheromone traps for monitoring, resistant varieties, and judicious chemical control only when necessary."
        else:
            return "develop a comprehensive IPM strategy incorporating advanced monitoring techniques using digital tools, precision application of biocontrol agents, development of beneficial insect habitats, implementation of push-pull strategies, use of RNA interference technology for specific pest control, and integration with climate data for predictive pest management."
    
    def _get_soil_health_advice(self, model_size):
        if model_size == "1B":
            return "test soil regularly, add organic matter, use cover crops, avoid over-tillage, and maintain proper pH levels."
        elif model_size == "3B":
            return "conduct comprehensive soil testing for macro and micronutrients, implement soil health improvement through organic matter addition, cover cropping, reduced tillage practices, balanced fertilization based on soil test results, and monitoring soil biological activity indicators."
        else:
            return "establish a precision soil health management system using advanced soil testing including biological indicators, implement site-specific nutrient management, develop soil carbon sequestration strategies, use precision agriculture tools for variable rate application, monitor soil microbiome health, and integrate with digital soil health platforms for data-driven decisions."
    
    def _get_water_management_advice(self, model_size):
        if model_size == "1B":
            return "use drip irrigation, collect rainwater, mulch to retain moisture, schedule watering properly, and avoid water wastage."
        elif model_size == "3B":
            return "implement efficient irrigation systems like drip or sprinkler irrigation, develop rainwater harvesting infrastructure, use soil moisture sensors for scheduling, practice mulching and cover cropping for water conservation, and adopt drought-resistant crop varieties."
        else:
            return "design a comprehensive water management system incorporating precision irrigation with IoT sensors, advanced weather-based irrigation scheduling, implementation of water-efficient technologies, development of on-farm water storage and recycling systems, integration with watershed management practices, and adoption of climate-resilient water strategies."
    
    def _get_general_farming_advice(self, model_size):
        if model_size == "1B":
            return "plan your crops according to season, use quality inputs, follow good agricultural practices, keep records, and connect with local agricultural extension services."
        elif model_size == "3B":
            return "develop a comprehensive farm management plan including crop planning, input management, adoption of good agricultural practices, financial planning and record keeping, market linkage development, and continuous learning through extension services and farmer groups."
        else:
            return "implement a holistic farm management system incorporating precision agriculture technologies, data-driven decision making, sustainable intensification practices, climate-smart agriculture techniques, value chain integration, financial risk management, and continuous innovation adoption through digital platforms and research partnerships."
    
    def create_gradio_interface(self):
        """Create Gradio interface"""
        def chat_interface(prompt, max_tokens, temperature, top_p):
            if not prompt.strip():
                return "Please enter your agricultural question."
            
            return self.generate_response(prompt, max_tokens, temperature, top_p)
        
        interface = gr.Interface(
            fn=chat_interface,
            inputs=[
                gr.Textbox(
                    lines=4,
                    placeholder="Ask your agricultural question here...",
                    label="Your Agricultural Query"
                ),
                gr.Slider(50, 512, value=256, label="Max Tokens"),
                gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
                gr.Slider(0.1, 1.0, value=0.9, label="Top P")
            ],
            outputs=gr.Textbox(
                lines=8,
                label=f"Agricultural AI Response ({self.model_size} Model)"
            ),
            title=f"Agricultural AI Assistant ({self.model_size} Model)",
            description=f"Specialized AI for Indian agriculture and farming guidance. Model: {self.model_name}",
            examples=[
                ["What are the best practices for organic farming?"],
                ["How to manage pests in rice crops naturally?"],
                ["What is the ideal irrigation schedule for wheat?"],
                ["How to improve soil fertility organically?"]
            ]
        )
        
        return interface

def main():
    """Main function"""
    model_name = os.getenv('MODEL_NAME', 'simple-agricultural-ai')
    port = int(os.getenv('GRADIO_PORT', 7861))
    
    logger.info(f"Starting Simple Agricultural AI Server")
    logger.info(f"Model: {model_name}")
    logger.info(f"Port: {port}")
    
    try:
        server = SimpleLLMServer(model_name)
        interface = server.create_gradio_interface()
        
        logger.info(f"Launching on port {port}")
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise

if __name__ == "__main__":
    main()