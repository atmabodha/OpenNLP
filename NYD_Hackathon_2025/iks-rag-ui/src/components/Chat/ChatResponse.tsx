import { FC } from 'react';
import ReactMarkdown from 'react-markdown';
import { useTypewriter } from '@/hooks/useTypewriter';

interface Reference {
    verse: string;
    text: string;
}

interface ChatResponseProps {
    summary?: string;
    explanation?: string;
    references?: Reference[];
    error?: boolean;
}

const ChatResponse: FC<ChatResponseProps> = ({ summary, explanation, references, error }) => {
    // Start summary immediately
    const { displayedText: displayedSummary, isComplete: isSummaryComplete } = useTypewriter(summary || '', 15);
    
    // Only start explanation after summary is complete
    const { displayedText: displayedExplanation, isComplete: isExplanationComplete } = useTypewriter(
        explanation || '', 
        10, 
        !isSummaryComplete && !!summary // Delay only if there's a summary and it's not complete
    );

    // Determine when to show references
    const showReferences = (!summary && !explanation) || // Show immediately if no streaming content
                         (summary && !explanation && isSummaryComplete) || // Show after summary if no explanation
                         (summary && explanation && isExplanationComplete); // Show after both are complete

    if (error) {
        return (
            <div className="max-w-4xl mx-auto">
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg overflow-hidden border border-red-200 dark:border-red-800">
                    <div className="p-6 flex items-center space-x-3">
                        <div className="h-10 w-10 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center">
                            <span className="text-xl">⚠️</span>
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-red-600 dark:text-red-400">Response Error</h2>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                                I apologize, but I encountered an error processing the response. Please try again.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                {/* Assistant Header */}
                <div className="border-b border-gray-100 dark:border-gray-700">
                    <div className="px-6 py-4 flex items-center space-x-3">
                        <div className="flex-shrink-0">
                            <div className="h-10 w-10 bg-indigo-100 dark:bg-indigo-900/50 rounded-full flex items-center justify-center">
                                <span className="text-xl">🤖</span>
                            </div>
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Assistant</h2>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Spiritual Guide</p>
                        </div>
                    </div>
                </div>
                
                {/* Main Content */}
                <div className="px-6 py-6 space-y-8">
                    {/* Summary Section */}
                    {summary && displayedSummary && (
                        <div className="animate-fade-in">
                            <div className="flex items-center space-x-2 mb-4">
                                <div className="h-8 w-8 bg-amber-100 dark:bg-amber-900/50 rounded-lg flex items-center justify-center">
                                    <span className="text-lg">⭐</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Summary</h3>
                            </div>
                            <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-5 border border-amber-100 dark:border-amber-800">
                                <div className="prose dark:prose-invert prose-amber prose-sm sm:prose-base max-w-none">
                                    <ReactMarkdown>
                                        {displayedSummary}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Detailed Answer Section */}
                    {explanation && displayedExplanation && (
                        <div className="animate-fade-in">
                            <div className="flex items-center space-x-2 mb-4">
                                <div className="h-8 w-8 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex items-center justify-center">
                                    <span className="text-lg">📚</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Detailed Explanation</h3>
                            </div>
                            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-5 border border-blue-100 dark:border-blue-800">
                                <div className="prose dark:prose-invert prose-blue prose-sm sm:prose-base max-w-none
                                            prose-headings:text-blue-900 dark:prose-headings:text-blue-100
                                            prose-p:text-gray-700 dark:prose-p:text-gray-300
                                            prose-strong:text-blue-700 dark:prose-strong:text-blue-300
                                            prose-ul:list-disc prose-ul:pl-5
                                            prose-li:text-gray-700 dark:prose-li:text-gray-300
                                            prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline">
                                    <ReactMarkdown>
                                        {displayedExplanation}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* References Section */}
                    {references && references.length > 0 && showReferences && (
                        <div className="animate-fade-in">
                            <div className="flex items-center space-x-2 mb-4">
                                <div className="h-8 w-8 bg-purple-100 dark:bg-purple-900/50 rounded-lg flex items-center justify-center">
                                    <span className="text-lg">💬</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">References</h3>
                            </div>
                            <div className="space-y-4">
                                {references.map((reference, index) => (
                                    <div key={index} 
                                        className="group bg-white dark:bg-gray-800 rounded-xl p-5 
                                                 border border-gray-200 dark:border-gray-700 
                                                 hover:border-purple-200 dark:hover:border-purple-700 
                                                 transition-all duration-300 hover:shadow-md"
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium 
                                                           bg-purple-100 dark:bg-purple-900/50 
                                                           text-purple-800 dark:text-purple-300"
                                            >
                                                <span className="mr-2">📖</span>
                                                {reference.verse}
                                            </span>
                                        </div>
                                        <p className="text-gray-600 dark:text-gray-300 italic">
                                            {reference.text}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatResponse; 