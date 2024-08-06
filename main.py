from transformers import MarianMTModel, MarianTokenizer

# Carregar o tokenizer e o modelo pré-treinados
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# Frase de origem para tradução
source = "I want to buy a car"
# source = "I'm a student"
# source = "Linear Algebra is awesome"

# Criar ids dos vetores de entrada codificados
input_ids = tokenizer(source, return_tensors="pt").input_ids

# Traduzir a frase
output_ids = model.generate(input_ids)[0]

# Decodificar e imprimir a tradução
target = tokenizer.decode(output_ids, skip_special_tokens=True)
print("en->de")
print("source:", source)
print("target:", target)
