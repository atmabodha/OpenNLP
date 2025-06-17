export interface Reference {
  source: string;
  link: string;
  text?: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  references?: Reference[];
  confidence?: number;
}

export interface AutoCompleteResponse {
  suggestions: string[];
}

export interface QueryPredictionResponse {
  predicted_query: string;
  confidence: number;
}

export interface ChatResponse {
  response: string;
  references: Reference[];
  confidence: number;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  suggestions: string[];
  predictedQuery?: string;
} 