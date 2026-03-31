import httpx
from openai import OpenAI
from client_v1.settings import settings

# Retrieval client
client = httpx.Client(
    base_url=settings.API_BASE,
    headers={"Authorization": f"Bearer {settings.API_KEY.get_secret_value()}"},
)


# LLM clients
client1 = OpenAI(
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    base_url=settings.OPENAI_API_BASE_URL,
)




















