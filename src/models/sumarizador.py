from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from model_initializers import initialize_pegasus
import torch
import logging
from dotenv import load_dotenv


load_dotenv()

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

src_text = [
    """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
]

model = initialize_pegasus()
result = model(src_text)[0].get("summary_text")
print(result)