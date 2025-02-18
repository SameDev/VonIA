import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# Carregar o dataset
df = pd.read_csv("dataset.csv")

# Remover espaços extras nas colunas
df.columns = df.columns.str.strip()

# Verificar novamente os nomes das colunas
print(df.columns)

# Usar um tokenizador pré-treinado para agilizar
tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")

# Adicionar um token de padding manualmente, caso não exista
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenizar perguntas e respostas
def tokenize_text(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    return encoding['input_ids']  # Retorna apenas os ids dos tokens

df["pergunta_tokens"] = df["pergunta"].apply(tokenize_text)
df["resposta_tokens"] = df["resposta"].apply(tokenize_text)

print("Dados carregados e tokenizados!")

# Carregar o modelo GPT-2 pré-treinado
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Modelo GPT-2 carregado!")

# Ajuste na função de geração de resposta
def generate_response(prompt, max_length=150):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # Gerar a saída usando o método generate do GPT-2
    output = model.generate(inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    # Decodificar a resposta
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Teste
print(generate_response("O que é um algoritmo?"))


# TODO: Treinar mais o modelo e criar prompts básicos, também intregar ao um db para exportação de treinamento e criação de dados.