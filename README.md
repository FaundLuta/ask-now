# Ask Now

Ask Now is a focused AI workspace powered by a Vercel Python function and the Groq API.

## Local run

Install dependencies and start the Flask app:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chatbot.py
```

Open `http://localhost:5000`.

## Vercel deployment

1. Import this repository into Vercel.
2. Add `GROQ_API_KEY` in Project Settings > Environment Variables. Use the same key as the local `.env`, but never commit `.env`.
3. Optionally add `GROQ_MODEL` and `GROQ_VISION_MODEL` to choose models.
4. Deploy. `vercel.json` routes the app through `api/index.py`.

The browser handles the dashboard, settings, local session history, voice input, and print-to-PDF export. Settings queries Groq's models endpoint and only lists text-output models accessible to the configured API key. Vision-capable models are marked in the picker; select one before attaching an image. The API extracts text from PDFs and validates image model compatibility.

## Security

`.env`, virtual environments, Python caches, and Vercel metadata are ignored by Git. If an API key has ever been exposed publicly, rotate it in the provider dashboard before deploying.
