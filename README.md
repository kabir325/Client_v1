# AI Load Balancer Client (Laptop)

## What This Does
This is the **client program** that runs on your laptops. It connects to the Raspberry Pi server, receives agricultural questions, processes them using local LLM models, and sends back responses.

## How to Run
```bash
cd Client_v1
python ai_load_balancer_client.py --server YOUR_RASPBERRY_PI_IP:50051
```

Example:
```bash
python ai_load_balancer_client.py --server 192.168.1.32:50051
```

## What Happens Step by Step

### 1. **Client Startup**
- 🚀 "AI Load Balancer Client v1.0 initialized"
- 🆔 Generates unique client ID (e.g., client-laptop-001)
- 🌐 Shows server address it will connect to

### 2. **System Analysis**
- 🖥️ Scans your laptop's hardware:
  - **CPU**: Cores and frequency
  - **RAM**: Total memory
  - **GPU**: Graphics card and memory
  - **Performance Score**: Calculated based on specs
- 📊 Shows: "AI Performance Score: 85.5"

### 3. **Server Connection**
- 🔌 "Connecting to AI Load Balancer at 192.168.1.32:50051"
- ✅ "Connected to AI Load Balancer successfully"
- 📡 Detects your laptop's local IP address

### 4. **Registration**
- 📝 Sends your laptop specs to Raspberry Pi server
- ✅ "Successfully registered as AI client"
- 🎯 Server assigns LLM model based on your performance:
  - **High specs**: Gets 8B model (best quality)
  - **Medium specs**: Gets 3B model (good quality)
  - **Low specs**: Gets 1B model (fast response)

### 5. **LLM Model Setup**
- 🤖 "Assigned LLM model: dhenu2-llama3.2-3b"
- 🌐 "Port: 7862"
- 🚀 Starts local LLM server on assigned port
- ✅ "Mock LLM server is ready!"

### 6. **Ready State**
- 💓 Starts heartbeat (keeps connection alive)
- 🌐 Starts gRPC server to receive requests
- ✅ "AI Client ready for distributed LLM processing"
- ⏳ Waits for questions from Raspberry Pi

### 7. **Processing Requests** (when server sends questions)
- 📥 "Received AI request: what is fertilizer?"
- 🎯 Shows assigned model and full prompt
- 📊 Shows your system specs again

### 8. **LLM Processing**
- 🔄 "Starting LLM processing with dhenu2-llama3.2-3b"
- 📡 Calls local LLM server on assigned port
- 🤖 LLM generates agricultural response
- ✅ "LLM processing completed"

### 9. **Response Delivery**
- 📤 Sends response back to Raspberry Pi server
- ⏱️ Shows processing time (e.g., "3.2 seconds")
- ✅ "Response sent successfully"

### 10. **Continuous Operation**
- 🔄 Stays connected and ready for more questions
- 💓 Sends periodic heartbeats to server
- 📱 Processes multiple requests as they come

## What Your Laptop Contributes
- **Processing Power**: Your laptop's CPU/GPU processes the LLM
- **Model Quality**: Better specs = better model = higher quality responses
- **Parallel Processing**: Multiple laptops work simultaneously
- **Load Distribution**: Shares the computational workload

## Logs You'll See
```
[CLIENT DATA FLOW] RECEIVED AI REQUEST
[CLIENT DATA FLOW] System Performance Score: 85.5
[DATA FLOW] Starting LLM processing
[DATA FLOW] Calling LLM server at localhost:7862
[DATA FLOW] LLM generated response
[CLIENT DATA FLOW] LLM PROCESSING COMPLETED
```

## Requirements
- Python 3.7+
- gRPC libraries
- gradio_client library
- Network connection to Raspberry Pi

## Your Role in the System
1. **Connect**: Your laptop joins the distributed network
2. **Contribute**: Your hardware processes AI requests
3. **Collaborate**: Works with other laptops for better results
4. **Respond**: Sends agricultural advice back to users

The client runs automatically once started - it will process requests and contribute to the distributed AI system! 🚀