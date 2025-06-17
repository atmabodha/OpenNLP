"use client";

import { useState, useRef, useEffect } from "react";
import debounce from "lodash/debounce";
import { motion } from "framer-motion";

interface ChatInputProps {
  onSend: (message: string) => void;
  onInputChange: (input: string) => void;
  isLoading: boolean;
  value: string;
  className?: string;
  clearInput: () => void;
}

export default function ChatInput({ onSend, onInputChange, isLoading, value, className, clearInput }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const debouncedInputChange = useRef(
    debounce((val: string) => {
      onInputChange(val);
    }, 300)
  ).current;

  useEffect(() => {
    if (value) {
      setMessage(value);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setMessage(newValue);
    debouncedInputChange(newValue);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message);
      setMessage("");
      debouncedInputChange("");
      clearInput();
    }
  };

  return (
    <form onSubmit={handleSubmit} className={`flex gap-3 bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-2xl shadow-lg p-3 ${className}`}>
      <textarea
        ref={textareaRef}
        value={message}
        onChange={handleChange}
        placeholder="Ask about ancient wisdom..."
        disabled={isLoading}
        rows={1}
        className="flex-1 p-4 rounded-xl border border-gray-200 dark:border-gray-700 
                   bg-white dark:bg-gray-800 resize-none focus:outline-none 
                   focus:ring-2 focus:ring-indigo-500/50 transition-all duration-200
                   text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500"
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
      />
      <button
        type="submit"
        disabled={isLoading || !message.trim()}
        className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl 
                   disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200
                   font-medium shadow-md hover:shadow-lg active:scale-95"
      >
        {isLoading ? (
          <motion.div 
            animate={{ rotate: 360 }} 
            transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
          />
        ) : (
          "Send"
        )}
      </button>
    </form>
  );
}
