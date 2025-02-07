import React, { useState, useEffect, useRef } from 'react';
import { Settings, Send, Search, Menu, X, ChevronDown } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxTokens: 2000,
    includeWebSearch: false,
    numSearchResults: 3
  });
  const [showSettings, setShowSettings] = useState(false);
  const [showModelSelect, setShowModelSelect] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    fetchModels();
    scrollToBottom();
  }, [messages]);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const selectModel = async (modelName) => {
    try {
      const response = await fetch('http://localhost:8000/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await response.json();
      setCurrentModel(modelName);
      setShowModelSelect(false);
    } catch (error) {
      console.error('Error selecting model:', error);
    }
  };

  const sendMessage = async () => {
        if (!input.trim() || !currentModel) return;

        const userMessage = {
        role: 'user',
        content: input
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
            prompt: input,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,           // Changed from maxTokens
            include_web_search: settings.includeWebSearch,  // Changed from includeWebSearch
            num_search_results: settings.numSearchResults   // Changed from numSearchResults
            })
        });

        const data = await response.json();
        
        const assistantMessage = {
            role: 'assistant',
            content: data.response,
            webSearchResults: data.web_search_results
        };

        setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
        console.error('Error sending message:', error);
        } finally {
        setLoading(false);
        }
    };

  return (
    <div className="flex flex-col h-screen bg-slate-100">
      {/* Header */}
      <div className="bg-white shadow-md p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button 
            onClick={() => setShowModelSelect(!showModelSelect)}
            className="flex items-center space-x-2 px-4 py-2 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors duration-200"
          >
            <span className="font-medium">{currentModel || 'Select Model'}</span>
            <ChevronDown size={16} />
          </button>
        </div>
        <button 
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 hover:bg-slate-100 rounded-full transition-colors duration-200"
        >
          <Settings size={20} />
        </button>
      </div>

      {/* Model Selection Dropdown */}
      {showModelSelect && (
        <div className="absolute top-16 left-4 bg-white shadow-lg rounded-lg p-2 z-20 min-w-[200px] border border-slate-200">
          {models.map((model) => (
            <button
              key={model.name}
              onClick={() => selectModel(model.name)}
              className="block w-full text-left px-4 py-2 hover:bg-slate-100 rounded transition-colors duration-200"
            >
              {model.name}
            </button>
          ))}
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-30 z-30 flex items-center justify-center">
          <div className="bg-white rounded-xl p-6 w-96 max-w-[90vw] shadow-xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-semibold">Settings</h3>
              <button 
                onClick={() => setShowSettings(false)}
                className="p-1 hover:bg-slate-100 rounded-full transition-colors duration-200"
              >
                <X size={20} />
              </button>
            </div>
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">Temperature: {settings.temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => setSettings(prev => ({...prev, temperature: parseFloat(e.target.value)}))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Max Tokens</label>
                <input
                  type="number"
                  value={settings.maxTokens}
                  onChange={(e) => setSettings(prev => ({...prev, maxTokens: parseInt(e.target.value)}))}
                  className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-50 rounded-lg">
                <label className="text-sm font-medium">Include Web Search</label>
                <div className="relative inline-block w-10 mr-2 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={settings.includeWebSearch}
                    onChange={(e) => setSettings(prev => ({...prev, includeWebSearch: e.target.checked}))}
                    className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
                </div>
              </div>
              {settings.includeWebSearch && (
                <div>
                  <label className="block text-sm font-medium mb-2">Number of Search Results</label>
                  <input
                    type="number"
                    value={settings.numSearchResults}
                    onChange={(e) => setSettings(prev => ({...prev, numSearchResults: parseInt(e.target.value)}))}
                    className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1"
                    max="10"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white ml-4'
                  : 'bg-white mr-4'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              {message.webSearchResults && (
                <div className="mt-3 pt-3 border-t border-opacity-20">
                  <div className="font-medium mb-1 text-sm">Web Search Results:</div>
                  <div className="text-sm opacity-90">{message.webSearchResults}</div>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white shadow-sm p-4 rounded-2xl max-w-[80%] mr-4">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t bg-white p-4 shadow-lg">
        <div className="max-w-4xl mx-auto flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type a message..."
            className="flex-1 p-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-50"
          />
          <button
            onClick={sendMessage}
            disabled={!currentModel || loading}
            className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      {/* Add some custom styles for the toggle switch */}
      <style jsx>{`
        .toggle-checkbox:checked {
          right: 0;
          border-color: #3B82F6;
        }
        .toggle-checkbox:checked + .toggle-label {
          background-color: #3B82F6;
        }
        .toggle-checkbox {
          right: 0;
          transition: all 0.3s;
        }
        .toggle-label {
          transition: background-color 0.3s;
        }
      `}</style>
    </div>
  );
};

export default ChatInterface;