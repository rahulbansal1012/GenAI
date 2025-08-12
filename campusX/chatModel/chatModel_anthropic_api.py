from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatXAI(
    model="grok-4",
    temperature=0,
    max_tokens= 500,
)

results  = llm.invoke("Hi Can you tell me the capital of India")
print(results)
