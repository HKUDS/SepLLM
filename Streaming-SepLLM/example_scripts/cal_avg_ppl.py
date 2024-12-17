import torch

ctx_len = int(1024 * 20)

outputs_path = '../outputs/xxx' 


ppl_log_file = 'log.txt'
out_file_name = f"ppl_len{ctx_len}.txt"

nlls = []
with open(outputs_path+ppl_log_file, 'r') as f:
    for i in range(ctx_len):
        cur_ppl_str = f.readline()
        if  len(cur_ppl_str) <=0:
            print(f"err: for i: {i}, cur_ppl_str: {cur_ppl_str}")
            exit(0)
            continue
        
        neg_log_likelihood =  torch.tensor(float(cur_ppl_str))
        
        nlls.append(neg_log_likelihood)


ppl = torch.exp(torch.stack(nlls).mean())
print(f"ppl: {ppl.item()}")

with open(outputs_path + out_file_name, 'w') as fout:
    fout.write(f"{ppl.item()}\n")