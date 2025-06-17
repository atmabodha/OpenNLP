import { ChatMessage } from "@/types/chat";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"; // Uses env variable

async function fetchData(endpoint: string, body: object) {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error(`Error: ${response.status} ${response.statusText}`);
      return { error: `Failed to fetch ${endpoint}`, status: response.status };
    }

    return await response.json();
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    return { error: "Network error. Please check your connection." };
  }
}

export const mockApiService = {
  async getSuggestions(partialQuery: string) {
    const data = await fetchData("/v1/autocomplete", { partial_query: partialQuery });
    return data?.suggestions || [];
  },

  async getPredictedQuery(partialQuery: string) {
    const data = await fetchData("/v1/predict-query", { query: partialQuery });
    return data?.predicted_query || null;
  },

  async getChatResponse(query: string): Promise<ChatMessage> {
    await new Promise((resolve) => setTimeout(resolve, 1000)); // Simulate delay
    console.log("query", query);

    const data = await fetchData("/v1/chat", { query });

    return data.response
      ? {
          id: Date.now().toString(),
          role: "assistant",
          content: data.response,
          references: data.references || [],
          confidence: data.confidence || 0,
          timestamp: new Date(),
        }
      : {
          id: Date.now().toString(),
          role: "assistant",
          content: "⚠️ Something went wrong. Please try again.",
          references: [],
          confidence: 0,
          timestamp: new Date(),
        };
  },

  async getReferences(query: string) {
    const data = await fetchData("/v1/retrieve-references", { query });
    return data?.references || [];
  },

  async regenerateResponse(query: string) {
    const data = await fetchData("/v1/regenerate", { query });
    return data?.new_response || "⚠️ Could not regenerate response.";
  },
};

