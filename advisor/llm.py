import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


dotenv.load_dotenv()


# ============================================================
# COMMON LLM CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"
TEMPERATURE = 0

MODEL_GEMMA_4_31B_IT = "gemma-4-31b-it"
TEMPERATURE_GEMMA_4_31B_IT = 0

MODEL_GROQ_GPTOSS120B = "openai/gpt-oss-120b"
TEMPERATURE_GROQ_GPTOSS120B = 0


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

llm_gemma_4_31b_it = ChatGoogleGenerativeAI(
    model=MODEL_GEMMA_4_31B_IT,
    temperature=TEMPERATURE_GEMMA_4_31B_IT,
)

llm_gpt_oss_120b = ChatGroq(
    model=MODEL_GROQ_GPTOSS120B,
    temperature=TEMPERATURE_GROQ_GPTOSS120B,
)
