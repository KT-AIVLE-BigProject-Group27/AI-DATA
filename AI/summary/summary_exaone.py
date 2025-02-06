import os
import pandas as pd
import re
import random
import numpy as np
import sys
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath("AI/combine"))
from modularization_ver4 import article_summary_AI_ver2, summary_model, summary_tokenizer


model = summary_model
tokenizer = summary_tokenizer

test_1 = pd.read_csv('Data_Analysis/Data_ver2/summary_data/화장품_summary_data.csv')
# test_1['input']에 generate_response 함수를 순차적으로 적용하여 결과를 'result' 컬럼에 저장
article = test_1['input'][5]
result = article_summary_AI_ver2(article)
print(result)
