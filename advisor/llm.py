import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


dotenv.load_dotenv()


# ============================================================
# COMMON LLM CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"
TEMPERATURE = 0


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)




# ============================================================
# COMMON LLM CONFIGURATION
# ============================================================

MODEL_GROQ_GPTOSS120B = "openai/gpt-oss-120b"
TEMPERATURE_GROQ_GPTOSS120B = 0


# ============================================================
# LLM
# ============================================================

llm_GPT_OSS = ChatGroq(
    model=MODEL_GROQ_GPTOSS120B,
    temperature=TEMPERATURE_GROQ_GPTOSS120B,
)