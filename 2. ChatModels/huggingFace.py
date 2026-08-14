import warnings
import dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Loads HUGGINGFACEHUB_API_TOKEN from .env automatically
dotenv.load_dotenv()

warnings.filterwarnings("ignore")

# Initialize endpoint (automatically reads HUGGINGFACEHUB_API_TOKEN from env)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = "how to be a good boy"

response = model.invoke(prompt)

print(response.content)

# idiot not working, ignore it this is optional for our work.