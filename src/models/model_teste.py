from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.6, model='gpt-4o-mini')

def count_albums(artist):
    prompt = ChatPromptTemplate.from_template("Quantos álbuns {artist} já lançou em sua carreira? E quais são?")
    prompt_val = prompt.invoke({"artist": artist})
    output = llm.invoke(prompt_val)

    return StrOutputParser().invoke(output)

if __name__=="__main__":
    print('=========================================')
    print('Resposta direta do gpt-3.5-turbo:')
    print(count_albums('Taylor Swift'))
    print('=========================================')