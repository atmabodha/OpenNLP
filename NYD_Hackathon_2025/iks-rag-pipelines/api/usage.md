# API Setup and Usage Guide

## Setup Instructions

1. Navigate to the API directory:
```bash
cd iks-rag-pipelines/api/
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   - On Unix/MacOS:
   ```bash
   source venv/bin/activate
   ```

4. Start the FastAPI development server:
```bash
fastapi dev
```

## Testing the API

1. Access the Swagger UI documentation:
   - Open your browser and navigate to: http://127.0.0.1:8000/docs#/
   (or your localhost)
   - The interactive documentation allows you to test all endpoints directly

## Customizing Helper Functions

The API includes modular helper functions that can be customized for specific use cases:

- **Auto Complete**: 
  - Location: `helper/auto_complete.py`
  - Purpose: Implement NLP logic for query completion
  - Usage: Modify the completion logic while maintaining the existing class interface
  - Example:
    ```python
    class AutoComplete:
        def get_suggestions(self, query: str) -> List[str]:
            # Implement your custom NLP logic here
            pass
    ```

Note: Helper class instances are already integrated with endpoints, so modifications to the internal logic will be automatically reflected in the API responses.
