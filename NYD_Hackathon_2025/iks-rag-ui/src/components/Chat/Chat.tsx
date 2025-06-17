"use client";

import { useState } from "react";
import { ChatMessage } from "@/types/chat";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import QuerySuggestions from "./QuerySuggestions";
import { apiService } from "@/lib/apiService";
import { motion } from "framer-motion";


// Predefined categories for question suggestions
const suggestionCategories = [
  {
    icon: "🕉️",
    title: "Spiritual Concepts",
    questions: [
      "What is karma yoga?",
      "Explain the concept of dharma",
      "What are the four paths of yoga?",
    ],
  },
  {
    icon: "🧘",
    title: "Meditation & Practice",
    questions: [
      "How to start meditation practice?",
      "What are the benefits of pranayama?",
      "Guide me through basic meditation",
    ],
  },
  {
    icon: "📚",
    title: "Gita Teachings",
    questions: [
      "What does Gita say about duty?",
      "Explain Chapter 2 Verse 47",
      "Purpose of life according to Gita",
    ],
  },
  {
    icon: "🎯",
    title: "Life Application",
    questions: [
      "How to apply Gita in daily life?",
      "Managing stress through Gita",
      "Work-life balance in Gita",
    ],
  },
];

export default function Chat() {
  // State to store chat messages
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // State for autocomplete suggestions
  const [suggestions, setSuggestions] = useState<string[]>([]);
  // State to store the predicted query
  const [predictedQuery, setPredictedQuery] = useState<string | null>(null);
  // Loading state for API response
  const [isLoading, setIsLoading] = useState(false);

  /**
   * Handles sending user messages and getting bot responses.
   * @param content - The user-input message.
   */
  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return; // Ignore empty messages
  
    // Create user message object
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };
  
    // Update chat messages
    setMessages((prev) => [...prev, userMessage]);
  
    // Clear suggestions & input
    setSuggestions([]);
    setPredictedQuery(null);
  
    // Fetch response from API
    setIsLoading(true);
    try {
      const response = await apiService.getChatResponse(content);
      setMessages((prev) => [...prev, response]); // Append bot response
    } catch (error) {
      console.error("Error fetching response:", error);
    } finally {
      setIsLoading(false);
    }
  };
  
  /**
   * Handles input change and fetches suggestions + predicted query.
   * @param input - The text entered by the user.
   */
  const handleInputChange = async (input: string) => {
    if (!input.trim()) {
      setSuggestions([]);
      setPredictedQuery(null);
      return;
    }

    try {
      const [newSuggestions, predicted] = await Promise.all([
        apiService.getSuggestions(input),
        apiService.getPredictedQuery(input),
      ]);
      setSuggestions(newSuggestions);
      setPredictedQuery(predicted);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  /**
   * Renders the welcome screen with query suggestions.
   */
  const renderWelcomeScreen = () => (
    <div className="h-full flex flex-col items-center justify-center p-4 md:p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-6 mb-12"
      >
        <h3 className="text-2xl md:text-3xl font-semibold bg-gradient-to-r from-gray-800 to-gray-600 dark:from-gray-200 dark:to-gray-400 bg-clip-text text-transparent">
          Start Your Spiritual Journey
        </h3>
        <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Explore the wisdom of ancient texts through modern conversation.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4 gap-4 md:gap-6 w-full max-w-7xl">
        {suggestionCategories.map((category, idx) => (
          <motion.div
            key={category.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white/90 dark:bg-gray-800/90 rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all duration-300
              border border-gray-100 dark:border-gray-700 backdrop-blur-lg backdrop-filter
              hover:scale-105 hover:bg-white dark:hover:bg-gray-800"
          >
            <div className="flex items-center gap-4 mb-4">
              <span className="text-3xl transform transition-transform group-hover:scale-110">{category.icon}</span>
              <h4 className="text-lg font-semibold bg-gradient-to-r from-gray-800 to-gray-600 dark:from-gray-200 dark:to-gray-400 bg-clip-text text-transparent">
                {category.title}
              </h4>
            </div>
            <div className="space-y-2.5">
              {category.questions.map((question) => (
                <button
                  key={question}
                  onClick={() => handleSendMessage(question)}
                  className="w-full text-left px-4 py-3 text-sm text-gray-600 dark:text-gray-400 
                    hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-xl transition-all duration-200
                    hover:text-gray-900 dark:hover:text-white group flex items-center justify-between
                    hover:shadow-md"
                >
                  <span className="line-clamp-1 p-0">{question}</span>
                  <span className="opacity-0 group-hover:opacity-100 transition-opacity transform group-hover:translate-x-1">→</span>
                </button>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-gray-50 dark:bg-gray-900">
      {/* Main chat container */}
      <div className="flex-1 overflow-hidden relative">
        <div className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700">
          {messages.length === 0 ? (
            renderWelcomeScreen()
          ) : (
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 pb-32">
              <MessageList messages={messages} />
              {isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  className="flex items-center space-x-2 text-gray-500 dark:text-gray-400"
                >
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-sm">Thinking...</span>
                </motion.div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Fixed input section at bottom */}
      <div className="fixed bottom-0 right-0 left-0 sm:left-80 bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-4">
          <QuerySuggestions
            suggestions={suggestions}
            clearInput={() => setSuggestions([])}
            predictedQuery={predictedQuery || undefined}
            onSelect={handleSendMessage}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg"
          />
          <ChatInput 
            onSend={handleSendMessage} 
            clearInput={() => setSuggestions([])}
            onInputChange={handleInputChange} 
            isLoading={isLoading} 
            value=""
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg"
          />
        </div>
      </div>
    </div>
  );
}