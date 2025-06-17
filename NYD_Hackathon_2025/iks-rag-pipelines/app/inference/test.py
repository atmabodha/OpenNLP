from together import Together

client = Together(api_key="59eb1b0031a4590a398eeb5092b18e8003824f2bd082e1e286f7784d4cd725bd")

response = client.embeddings.create(
  model="BAAI/bge-base-en-v1.5",
  input="Our solar system orbits the Milky Way galaxy at about 515,000 mph",
)
print(len(response.data[0].embedding))