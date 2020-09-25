import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

text = """

The conventional alarm systems to detect fires, notify occupants, summon emergency responders, and provide information to help manage the response.  
Automatic Sprinkler system that detects the fire's heat, initiate alarm, and begin suppression within moments after flames appear 

"""


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: " + preprocess_text
print("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

# summarize
summary_ids = model.generate(tokenized_text,
                             num_beams=4,
                             no_repeat_ngram_size=2,
                             min_length=30,
                             max_length=100,
                             early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n\nSummarized text: \n", output)

# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.
