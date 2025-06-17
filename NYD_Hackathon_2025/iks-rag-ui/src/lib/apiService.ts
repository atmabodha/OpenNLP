// apiService.ts

import { ChatMessage } from "@/types/chat";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BaseApiResponse {
  error?: string;
}

interface PipelineResponse extends BaseApiResponse {
  response: string;
}

interface AutocompleteResponse extends BaseApiResponse {
  suggestions: string[];
}

interface QueryPredictionResponse extends BaseApiResponse {
  predicted_query: string;
  confidence: number;
}

async function fetchWithAuth<T>(endpoint: string, body: object): Promise<T> {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data as T;
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    throw error;
  }
}

export const apiService = {
  async getChatResponse(query: string): Promise<ChatMessage> {
    try {
      const data = await fetchWithAuth<PipelineResponse>("/v1/response-cache", { query });
      
      if (!data.response) {
        throw new Error("No response received from the server");
      }

      return {
        id: Date.now().toString(),
        role: "assistant",
        content: data.response,
        references: [], // Backend currently doesn't return references
        confidence: 1, // Backend currently doesn't return confidence
        timestamp: new Date(),
      };
    } catch (error) {
      console.error("Error getting chat response:", error);

      // Throw the error to let the component handle it
      throw new Error(error instanceof Error ? error.message : "Failed to get chat response");
    }
  },

  async getSuggestions(partialQuery: string): Promise<string[]> {
    try {
      const data = await fetchWithAuth<AutocompleteResponse>("/v1/recommendation", {
        query: partialQuery,
      });
      console.log(data.suggestions);
      return data.suggestions || [];
    } catch (error) {
      console.error("Error getting suggestions:", error);
      return [];
    }
  },

  async getPredictedQuery(partialQuery: string): Promise<string | null> {
    try {
      const data = await fetchWithAuth<QueryPredictionResponse>("/v1/predict-query", {
        query: partialQuery,
      });
      return data.predicted_query || null;
    } catch (error) {
      console.error("Error getting predicted query:", error);
      return null;
    }
  },

  // Helper method to check API status
  async checkStatus(): Promise<boolean> {
    try {
      const response = await fetch(`${BASE_URL}/status`);
      return response.ok;
    } catch (error) {
      console.error("Error checking API status:", error);
      return false;
    }
  }
};