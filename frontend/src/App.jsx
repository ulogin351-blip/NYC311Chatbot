import { useState, useRef, useEffect } from "react";
import "./App.css";
import ChartComponent from "./ChartComponent";
import DataTable from "./DataTable";

function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([
    { 
      role: "system", 
      content: "🏙️ Welcome to the NYC 311 AI Analytics Assistant! I can help you analyze NYC service request data with:\n\n📊 Interactive Visualizations (charts, graphs, maps)\n📈 Statistical Analysis & Insights\n📋 Data Tables & Summaries\n🔍 Intelligent Query Processing\n\nAsk me questions like:\n• \"Show me the top 5 complaint types\"\n• \"What percentage of noise complaints were resolved?\"\n• \"Create a chart of complaints by borough\"\n• \"Analyze response times for different agencies\"\n\nI'll provide detailed analysis with visualizations and actionable insights!" 
    }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const MAX_HISTORY_LENGTH = 20; // Maximum number of messages to keep in history

  // Scroll to bottom whenever chatHistory changes
  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, loading]);

  const scrollToBottom = () => {
    // Use setTimeout to ensure DOM has updated
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ 
        behavior: "smooth",
        block: "end",
        inline: "nearest"
      });
    }, 100);
  };
  
  // Function to trim history if it exceeds the maximum length
  const trimHistory = (history) => {
    if (history.length <= MAX_HISTORY_LENGTH) return history;
    
    // Always keep the system message if it exists
    const systemMessage = history.find(msg => msg.role === "system");
    let trimmedHistory = history.slice(-MAX_HISTORY_LENGTH);
    
    // If there was a system message and it's not in the trimmed history, add it back
    if (systemMessage && !trimmedHistory.some(msg => msg.role === "system")) {
      trimmedHistory = [systemMessage, ...trimmedHistory.slice(1)];
    }
    
    return trimmedHistory;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() || loading) return;

    // Add user message to chat
    const userMessage = message.trim();
    
    // Add the new user message to history
    setChatHistory((prev) => {
      const newHistory = [...prev, { role: "user", content: userMessage }];
      return trimHistory(newHistory);
    });
    
    // Clear input and set loading
    setMessage("");
    setLoading(true);

    try {
      // Use direct API URL (no proxy)
      const apiUrl = "http://localhost:8000/chat";
      console.log(`Sending direct request to ${apiUrl}`);
      
      // Prepare the conversation history to send
      const currentHistory = chatHistory.concat({ role: "user", content: userMessage });
      
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ 
          message: userMessage,
          conversation_history: currentHistory
        }),
        mode: "cors", // Enable CORS
      });
      
      console.log("Response status:", response.status);
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response data:", data);
      
      // Check if the response is a clarification request
      const isClarificationRequest = data.requires_clarification === true;
      const visualization = data.visualization || null;
      
      setChatHistory((prev) => {
        const newHistory = [
          ...prev,
          { 
            role: "assistant", 
            content: data.response || "Sorry, I couldn't process your request.",
            requires_clarification: isClarificationRequest,
            visualization: visualization
          }
        ];
        return trimHistory(newHistory);
      });
      setLoading(false);
    } catch (error) {
      console.error("Error:", error);
      setChatHistory((prev) => {
        const newHistory = [
          ...prev,
          { 
            role: "assistant", 
            content: "Sorry, there was an error processing your request. Please try again."
          }
        ];
        return trimHistory(newHistory);
      });
      setLoading(false);
    }
  };

  // Function to determine if content is HTML
  const isHTML = (str) => {
    return /<[a-z][\s\S]*>/i.test(str);
  };

  return (
    <div className="app-container">
      <header>
        <h1>NYC 311 AI Analytics Assistant</h1>
        <div className="header-subtitle">
          <p>Intelligent Data Analysis • Interactive Visualizations • AI-Powered Insights</p>
          <div className="capabilities-badges">
            <span className="capability-badge">📊 Charts & Graphs</span>
            <span className="capability-badge">📋 Data Tables</span>
            <span className="capability-badge">🔍 Smart Analytics</span>
            <span className="capability-badge">📈 Statistical Insights</span>
          </div>
        </div>
      </header>
      
      <div className="chat-container">
        <div className="chat-messages">
          {chatHistory.map((msg, index) => (
            <div 
              key={index} 
              className={`message ${msg.role} ${isHTML(msg.content) ? 'html-content' : ''} ${msg.requires_clarification ? 'clarification-request' : ''}`}
            >
              {msg.requires_clarification && (
                <div className="clarification-badge">Clarification Needed</div>
              )}
              {isHTML(msg.content) ? (
                <div dangerouslySetInnerHTML={{ __html: msg.content }} />
              ) : (
                msg.content
              )}
              {msg.visualization && (
                <div className="visualization-container">
                  {msg.visualization.type === 'table' ? (
                    <DataTable visualizationData={msg.visualization} />
                  ) : (
                    <ChartComponent visualizationData={msg.visualization} />
                  )}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant loading">
              <div className="loading-content">
                <span className="loading-dots">Analyzing data and generating insights</span>
                <div className="loading-description">
                  🔍 Processing your query • 📊 Preparing visualizations • 📈 Calculating statistics
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {chatHistory.length <= 1 && (
          <div className="example-queries">
            <p className="examples-title">💡 Try these example questions:</p>
            <div className="example-buttons">
              <button 
                className="example-btn"
                onClick={() => setMessage("Show me the top 5 complaint types with a bar chart")}
                disabled={loading}
              >
                📊 Top 5 complaint types chart
              </button>
              <button 
                className="example-btn"
                onClick={() => setMessage("What percentage of complaints were closed within 30 days?")}
                disabled={loading}
              >
                📈 Closure rate analysis
              </button>
              <button 
                className="example-btn"
                onClick={() => setMessage("Create a pie chart of complaints by borough")}
                disabled={loading}
              >
                🥧 Borough distribution chart
              </button>
              <button 
                className="example-btn"
                onClick={() => setMessage("Analyze response times for different agencies")}
                disabled={loading}
              >
                ⏱️ Response time analysis
              </button>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask for charts, data analysis, or insights about NYC 311 service requests..."
            disabled={loading}
          />
          <button type="submit" disabled={!message.trim() || loading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;

