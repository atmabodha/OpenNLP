import { useEffect, useRef } from "react";
import { ChatMessage } from "@/types/chat";

import { motion } from "framer-motion";

import ChatResponse from "./ChatResponse";

interface MessageListProps {
  messages: ChatMessage[];
}

interface ParsedResponse {
  summary_answer?: string;
  detailed_answer?: string;
  references?: Array<{
    source: string;
    chapter: string;
    verse: string;
    text: string;
  }>;
}

export default function MessageList({ messages }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const parseResponse = (parsed: ParsedResponse) => {
    try {
      return {
        summary: parsed.summary_answer,
        explanation: parsed.detailed_answer,
        references: parsed.references?.map((ref) => ({
          verse: `${ref.source} ${ref.chapter}:${ref.verse}`,
          text: ref.text
        })) || [],
        error: false
      };
    } catch (error) {
      console.error('Error parsing response:', error);
      return {
        error: true
      };
    }
  };

  return (
    <div className="space-y-6">
      {messages.map((message) => (
        <motion.div
          key={message.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
        >
          {message.role === "user" ? (
            <div className="max-w-[85%] md:max-w-[75%] rounded-2xl px-4 py-3 bg-indigo-600 text-white">
              <p className="prose prose-invert max-w-none prose-p:leading-relaxed">
                {message.content}
              </p>
            </div>
          ) : (
            <div className="max-w-[85%] md:max-w-[75%]">
              <ChatResponse {...parseResponse(message.content as ParsedResponse)} />
            </div>
          )}
        </motion.div>
      ))}
      {/* Empty div to scroll into view */}
      <div ref={messagesEndRef} />
    </div>
  );
}
