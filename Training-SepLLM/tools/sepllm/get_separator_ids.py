from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  # cache_dir="./pythia-70m-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  # cache_dir="./pythia-70m-deduped/step143000",
)
inputs_tokens = ['.', ',', '?', '!', ';', ":", ' ', '\t','\n',   
              '  ', ' .', ' ,', ' ?', ' !', ' ;', " :",  ' \t',' \n',
              '   ', '  .', '  ,', '  ?', '  !', '  ;', "  :",  '  \t', '  \n'  ,
              
              '. ', ', ', '? ', '! ', '; ', ": ", '\t ','\n ',
              '.  ',  ',  ',  '?  ',  '!  ',  ';  ',  ":  ", '\t  ',  '\n  '
              ]



inputs_ids = []
for i in range(len(inputs_tokens)):
    inputs_ids.append( tokenizer(inputs_tokens[i], return_tensors="pt") )


print("#######################For encoding#################################")
for i in range(len(inputs_ids)):
    print("_____________________")
    print(f"inputs_tokens: {inputs_tokens[i]}")
    print(f"inputs_ids: {inputs_ids[i]['input_ids']}")
    print("^^^^^^^^^^^^^^^^^^^^^^\n\n")


print("\n\n\n\n#######################For decoding#################################")
for i in range(len(inputs_ids)):
    print("___________Reverse Test__________")
    print(f"inputs_ids got before: {inputs_ids[i]['input_ids']}")    
    print(f"newly decoded: { tokenizer.decode( inputs_ids[i]['input_ids'][0] )  }")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")

