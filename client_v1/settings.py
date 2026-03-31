from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
import json

class EmmRetrieversSettings(BaseSettings):
    API_BASE: str
    API_KEY: SecretStr

    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: SecretStr

    LANGCHAIN_API_KEY: SecretStr

    DEFAULT_CLUSTER: str = "rag-os"
    DEFAULT_INDEX: str = "mine_e_emb-rag_live"

    DEFAULT_TIMEOUT: int = 120

    model_config = SettingsConfigDict(env_prefix="EMM_RETRIEVERS_", env_file="../.env")


EMM_API_URL="https://api.emm4u.eu/retrievers/v1"
with open('emm_token.json', 'r') as file:
    config = json.load(file)
    EMM_API_KEY = config['EMM_RETRIEVERS_API_KEY']
    
GPT_API_URL="https://api-gpt.jrc.ec.europa.eu/v1"
with open('gpt_token.json', 'r') as file:
    config = json.load(file)
    GPT_API_KEY = config['GPT_API_KEY']

LANCHAIN_API_URL="LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com"
with open('langchain-token.json', 'r') as file: 
    config = json.load(file)
    LANGCHAIN_API_KEY = config['LANGCHAIN_API_KEY']


settings = EmmRetrieversSettings(
        API_BASE=EMM_API_URL,
        API_KEY=EMM_API_KEY ,
        OPENAI_API_BASE_URL=GPT_API_URL,
        OPENAI_API_KEY=GPT_API_KEY,
        LANGCHAIN_API_KEY=LANGCHAIN_API_KEY
    )