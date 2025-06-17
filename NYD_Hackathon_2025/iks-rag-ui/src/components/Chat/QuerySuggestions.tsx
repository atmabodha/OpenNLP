"use client";

import { motion } from "framer-motion";

interface QuerySuggestionsProps {
  suggestions: string[];
  predictedQuery?: string;
  onSelect: (query: string) => void;
  className?: string;
  clearInput: () => void;  // New prop for clearing input
}

const capitalizeFirstLetter = (string: string) => {
  return string.charAt(0).toUpperCase() + string.slice(1).toLowerCase();
};

export default function QuerySuggestions({ 
  suggestions, 
  predictedQuery, 
  onSelect,
  className,
  clearInput
}: QuerySuggestionsProps) {
  if (!suggestions.length && !predictedQuery) return null;

  const handleSelect = (query: string) => {
    onSelect(query);
    clearInput(); // Clear input when suggestion is selected
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`space-y-4 ${className}`}  
    >
      {/* Predicted Query Suggestion */}
      {predictedQuery && (
        <div className="text-sm px-2">
          <span className="text-gray-500 dark:text-gray-400">Did you mean: </span>
          <button
            onClick={() => handleSelect(predictedQuery)}
            className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 
                     dark:hover:text-indigo-300 font-medium hover:underline 
                     transition-colors duration-200"
            aria-label={`Did you mean ${predictedQuery}?`}
          >
            {capitalizeFirstLetter(predictedQuery)}
          </button>
        </div>
      )}

      {/* Suggested Queries List */}
      {suggestions.length > 0 && (
        <div className="max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 
                      dark:scrollbar-thumb-gray-600 scrollbar-track-transparent">
          <div className="flex flex-col space-y-2 px-2">
            {suggestions.map((suggestion, index) => (
              <motion.button
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => handleSelect(suggestion)}
                className="text-left px-4 py-3 text-sm bg-gray-50 dark:bg-gray-700/50 
                         text-gray-700 dark:text-gray-300 rounded-xl
                         hover:bg-indigo-50 dark:hover:bg-indigo-900/30 
                         hover:text-indigo-700 dark:hover:text-indigo-300
                         transition-all duration-200 shadow-sm hover:shadow
                         border border-transparent hover:border-indigo-200 
                         dark:hover:border-indigo-800 transform hover:-translate-y-0.5
                         hover:scale-101 group"
                aria-label={`Select suggestion: ${suggestion}`}
              >
                <span className="group-hover:pl-1 transition-all duration-200">
                  {capitalizeFirstLetter(suggestion)}
                </span>
              </motion.button>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
