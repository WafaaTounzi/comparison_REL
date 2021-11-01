#!/usr/bin/env python
# coding: utf-8

# In[1]:


python


# In[2]:


import torch
x = torch.rand(5, 3)
print(x)


# In[3]:


pip install transformers[torch]


# In[4]:


python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"


# In[5]:


pip install transformers


# In[6]:


from transformers import pipeline
classifier = pipeline('sentiment-analysis')


# In[10]:


classifier('i am happy about this code')


# In[11]:


results = classifier(["its a pleasure to know data science.",
           "hope i don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


# In[13]:


from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[14]:


pip install tensorflow


# In[16]:


from transformers import BertModel, BertConfig

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config


# In[17]:


from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, is this working", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state


# In[18]:


from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits


# In[19]:


from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[22]:


inputs = tokenizer("is this working.")


# In[23]:


print(inputs)


# In[3]:


from transformers import pipeline
classifier = pipeline('sentiment-analysis')


# In[7]:


classifier('This code is not simple.')


# In[8]:


from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


# In[9]:


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


# In[17]:


classifier("je suis une travailleuse acharnée")


# In[18]:


classifier("Soy un poco trabajador")


# In[19]:


from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my code is working", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state


# In[20]:


print(outputs)


# In[ ]:




