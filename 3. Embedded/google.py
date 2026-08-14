from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
import dotenv
dotenv.load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature="2" )

prompt = """how to be a good boy"""

response = llm.invoke(prompt)

print(response.text)