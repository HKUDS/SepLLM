from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
inputs_str = [".", " .", " . ", "hello.", "\n", " \n", " \n ", "hello\n"]
inputs_ids = []
for i in range(len(inputs_str)):
    inputs_ids.append( tokenizer(inputs_str[i], return_tensors="pt") )
# inputs.append( tokenizer(".", return_tensors="pt") )
# inputs.append( tokenizer(" .", return_tensors="pt") )
# inputs.append( tokenizer(" . ", return_tensors="pt") )
# inputs.append( tokenizer("hello.", return_tensors="pt") )
# inputs.append( tokenizer("\n", return_tensors="pt") )
# inputs.append( tokenizer(" \n", return_tensors="pt") )
# inputs.append( tokenizer(" \n ", return_tensors="pt") )
# inputs.append( tokenizer("hello\n ", return_tensors="pt") )


out = []
print("For encode")
for i in range(len(inputs)):
    print("_____________________")
    print(f"inputs_str: {inputs_str[i]}")
    print(f"inputs_ids: {inputs_ids[i]}")
    print("^^^^^^^^^^^^^^^^^^^^^^")

# tokens = model.generate(**inputs)

print("#######################for decode#################################")
for i in range(len(inputs_ids)):
    print("_____________________")
    # print(f"inputs_str: {tokenizer.decode(inputs_ids[i])}")
    print(f"inputs_ids: {inputs_ids[i]}")
    print(f"decode: {tokenizer.decode(inputs_ids[i])}")
    print("^^^^^^^^^^^^^^^^^^^^^^")

# tokenizer.decode(tokens[0])