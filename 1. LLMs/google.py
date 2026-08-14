from langchain_google_genai import GoogleGenerativeAI
import warnings
import dotenv
dotenv.load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

llm = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = """how to be a good boy"""

response = llm.invoke(prompt)

print(response)