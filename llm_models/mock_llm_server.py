#!/usr/bin/env python3
"""
Mock LLM Server for Testing Distributed Load Balancer
This server simulates actual LLM responses for testing purposes
"""

import os
import gradio as gr
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockLLMServer:
    def __init__(self, model_name="mock-agricultural-ai"):
        self.model_name = model_name
        self.model_size = self._get_model_size(model_name)
        logger.info(f"[MOCK LLM] Initializing {model_name} ({self.model_size})")
        
    def _get_model_size(self, model_name):
        """Get model size from name"""
        if "1b" in model_name.lower():
            return "1B"
        elif "3b" in model_name.lower():
            return "3B"
        elif "8b" in model_name.lower():
            return "8B"
        else:
            return "Mock"
    
    def generate_response(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """Generate mock agricultural AI response"""
        try:
            logger.info(f"[MOCK LLM] Processing prompt: '{prompt[:50]}...'")
            logger.info(f"[MOCK LLM] Model: {self.model_name} ({self.model_size})")
            logger.info(f"[MOCK LLM] Parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
            
            # Simulate realistic processing time based on model size
            if self.model_size == "1B":
                processing_time = 1.0 + random.uniform(0.5, 1.0)  # 1.5-2.0 seconds
            elif self.model_size == "3B":
                processing_time = 2.0 + random.uniform(1.0, 2.0)  # 3.0-4.0 seconds
            elif self.model_size == "8B":
                processing_time = 4.0 + random.uniform(2.0, 3.0)  # 6.0-7.0 seconds
            else:
                processing_time = 1.5
            
            logger.info(f"[MOCK LLM] Simulating {processing_time:.1f}s processing time...")
            time.sleep(processing_time)
            
            # Generate contextual agricultural response based on model size
            response = self._generate_agricultural_response(prompt, self.model_size)
            
            logger.info(f"[MOCK LLM] Generated response: {len(response)} chars")
            logger.info(f"[MOCK LLM] Response preview: '{response[:100]}...'")
            
            return response
            
        except Exception as e:
            logger.error(f"[MOCK LLM] Response generation failed: {e}")
            raise Exception(f"Mock LLM processing failed: {e}")
    
    def _generate_agricultural_response(self, prompt, model_size):
        """Generate contextual agricultural response based on prompt and model size"""
        
        # Analyze prompt for agricultural context
        prompt_lower = prompt.lower()
        
        # Generate different quality responses based on model size
        if model_size == "1B":
            response_quality = "basic"
            model_prefix = f"[{self.model_name} - 1B Model Response]"
        elif model_size == "3B":
            response_quality = "detailed"
            model_prefix = f"[{self.model_name} - 3B Model Response]"
        elif model_size == "8B":
            response_quality = "expert"
            model_prefix = f"[{self.model_name} - 8B Model Response]"
        else:
            response_quality = "basic"
            model_prefix = f"[{self.model_name} - Mock Model Response]"
        
        # Context-specific responses
        if any(word in prompt_lower for word in ['fertilizer', 'nutrient', 'feed']):
            topic = "fertilizers and nutrients"
            content = self._get_fertilizer_advice(response_quality)
        elif any(word in prompt_lower for word in ['organic', 'natural', 'pesticide-free']):
            topic = "organic farming"
            content = self._get_organic_farming_advice(response_quality)
        elif any(word in prompt_lower for word in ['rice', 'paddy', 'grain']):
            topic = "rice cultivation"
            content = self._get_rice_farming_advice(response_quality)
        elif any(word in prompt_lower for word in ['pest', 'insect', 'disease', 'bug']):
            topic = "pest management"
            content = self._get_pest_management_advice(response_quality)
        elif any(word in prompt_lower for word in ['soil', 'fertility', 'nutrient']):
            topic = "soil health"
            content = self._get_soil_health_advice(response_quality)
        elif any(word in prompt_lower for word in ['water', 'irrigation', 'drought']):
            topic = "water management"
            content = self._get_water_management_advice(response_quality)
        else:
            topic = "general farming"
            content = self._get_general_farming_advice(response_quality)
        
        return f"{model_prefix} Regarding {topic}: {content}"
    
    def _get_fertilizer_advice(self, quality):
        if quality == "basic":
            return "Fertilizers provide essential nutrients (NPK - Nitrogen, Phosphorus, Potassium) to plants. Use balanced fertilizers based on soil testing. Apply organic compost and chemical fertilizers as needed for your crop type."
        elif quality == "detailed":
            return "Fertilizers are crucial for crop nutrition. Conduct soil testing to determine NPK ratios. Use organic fertilizers like compost, vermicompost, and green manure for long-term soil health. Apply chemical fertilizers in split doses during critical growth stages. Consider micronutrients like zinc, iron, and boron based on deficiency symptoms."
        else:
            return "Comprehensive fertilizer management involves soil testing, nutrient budgeting, and precision application. Implement integrated nutrient management combining organic sources (FYM, compost, biofertilizers) with inorganic fertilizers. Use slow-release fertilizers and foliar applications for efficiency. Monitor plant tissue analysis and adjust fertilization programs based on crop response and environmental conditions."
    
    def _get_organic_farming_advice(self, quality):
        if quality == "basic":
            return "Use natural fertilizers like compost and cow dung, avoid chemical pesticides, practice crop rotation, and use neem oil for pest control."
        elif quality == "detailed":
            return "Implement integrated organic practices including composting, green manuring, biological pest control using beneficial insects, crop diversification, soil health management through organic matter addition, and consider certification processes for premium market access."
        else:
            return "Develop comprehensive organic farming systems incorporating biodynamic principles, precision composting techniques, advanced biocontrol strategies, soil microbiome enhancement, carbon sequestration practices, sustainable supply chain development, and comprehensive certification for international market access."
    
    def _get_rice_farming_advice(self, quality):
        if quality == "basic":
            return "Prepare fields properly, use quality seeds, maintain proper water levels, apply fertilizers at right time, and control weeds regularly."
        elif quality == "detailed":
            return "Follow System of Rice Intensification (SRI) methods, ensure proper land preparation with laser leveling, use certified seeds with appropriate spacing, implement alternate wetting and drying for water management, and integrate pest management strategies."
        else:
            return "Implement climate-smart rice cultivation using drought-tolerant varieties, precision agriculture techniques, real-time monitoring systems, optimized nutrient management based on soil sensors, integrated water management, mechanization for efficiency, and post-harvest quality management."
    
    def _get_pest_management_advice(self, quality):
        if quality == "basic":
            return "Identify pests early, use neem-based sprays, encourage beneficial insects, remove infected plants, and follow crop rotation."
        elif quality == "detailed":
            return "Implement Integrated Pest Management (IPM) combining cultural practices, biological control using parasitoids and predators, selective use of biopesticides, pheromone traps for monitoring, and resistant varieties."
        else:
            return "Develop comprehensive IPM strategies incorporating advanced monitoring using digital tools, precision application of biocontrol agents, beneficial insect habitat development, push-pull strategies, and integration with climate data for predictive pest management."
    
    def _get_soil_health_advice(self, quality):
        if quality == "basic":
            return "Test soil regularly, add organic matter, use cover crops, avoid over-tillage, and maintain proper pH levels."
        elif quality == "detailed":
            return "Conduct comprehensive soil testing for macro and micronutrients, implement soil health improvement through organic matter addition, cover cropping, reduced tillage practices, and monitor soil biological activity indicators."
        else:
            return "Establish precision soil health management systems using advanced soil testing including biological indicators, implement site-specific nutrient management, develop soil carbon sequestration strategies, and integrate with digital soil health platforms for data-driven decisions."
    
    def _get_water_management_advice(self, quality):
        if quality == "basic":
            return "Use drip irrigation, collect rainwater, mulch to retain moisture, schedule watering properly, and avoid water wastage."
        elif quality == "detailed":
            return "Implement efficient irrigation systems like drip or sprinkler irrigation, develop rainwater harvesting infrastructure, use soil moisture sensors for scheduling, and adopt drought-resistant crop varieties."
        else:
            return "Design comprehensive water management systems incorporating precision irrigation with IoT sensors, advanced weather-based irrigation scheduling, implementation of water-efficient technologies, and integration with watershed management practices."
    
    def _get_general_farming_advice(self, quality):
        if quality == "basic":
            return "Plan your crops according to season, use quality inputs, follow good agricultural practices, keep records, and connect with local agricultural extension services."
        elif quality == "detailed":
            return "Develop comprehensive farm management plans including crop planning, input management, adoption of good agricultural practices, financial planning and record keeping, and market linkage development."
        else:
            return "Implement holistic farm management systems incorporating precision agriculture technologies, data-driven decision making, sustainable intensification practices, climate-smart agriculture techniques, and value chain integration."
    
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
            description=f"Mock LLM for testing distributed load balancer. Model: {self.model_name}",
            examples=[
                ["What is fertilizer and how to use it?"],
                ["Best practices for organic farming?"],
                ["How to manage pests in rice crops?"],
                ["Water management techniques for farming?"]
            ]
        )
        
        return interface

def main():
    """Main function"""
    model_name = os.getenv('MODEL_NAME', 'mock-agricultural-ai')
    port = int(os.getenv('GRADIO_PORT', 7861))
    
    logger.info(f"Starting Mock Agricultural AI Server")
    logger.info(f"Model: {model_name}")
    logger.info(f"Port: {port}")
    
    try:
        server = MockLLMServer(model_name)
        interface = server.create_gradio_interface()
        
        logger.info(f"Launching on port {port}")
        
        # Try to launch with automatic port finding if the specified port is busy
        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True
            )
        except OSError as e:
            if "Cannot find empty port" in str(e):
                logger.warning(f"Port {port} is busy, trying nearby ports...")
                # Try a few nearby ports
                for alternative_port in range(port + 1, port + 10):
                    try:
                        logger.info(f"Trying alternative port {alternative_port}")
                        interface.launch(
                            server_name="0.0.0.0",
                            server_port=alternative_port,
                            share=False,
                            show_error=True
                        )
                        logger.info(f"Successfully launched on alternative port {alternative_port}")
                        break
                    except OSError:
                        continue
                else:
                    logger.error("Could not find any available port")
                    raise e
            else:
                raise e
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise

if __name__ == "__main__":
    main()