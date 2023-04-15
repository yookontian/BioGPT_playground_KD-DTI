pred_file = 'analysis/predT_not_in_original_pmid_predTs.json'
gold_file = 'data/KD-DTI/raw/test.json'

import json

# load the pmids that the pred_Ts are not in the original articles
with open (pred_file, 'r') as f:
    pred_d_not_in_original = json.load(f)


# load the gold standard
with open (gold_file, 'r') as f:
    gold_d = json.load(f)



import torch
from src.transformer_lm_prompt import TransformerLanguageModelPrompt
m = TransformerLanguageModelPrompt.from_pretrained(
        "checkpoints/RE-DTI-BioGPT", 
        "checkpoint_avg.pt", 
        "data/KD-DTI/relis-bin",
        tokenizer='moses', 
        bpe='fastbpe', 
        bpe_codes="data/bpecodes",
        max_len_b=1024,
        beam=5)
m.cuda()

# because it's hard to use the moses tokenizer.decode() to show the different between the 4 (= 4#) and 4</w>
# so here using the tokenizer from HF to decode each generated token, and it doesn't include the learn0 - learn9, which is not a problem
from transformers import BioGptTokenizer

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

def get_test_data(id):
    prefix = torch.arange(42384, 42393)
    test_data = {}
    test_data['pmid'] = pred_d_not_in_original[id]['id']
    test_data['text'] = gold_d[test_data['pmid']]['title'].strip() + " " + gold_d[test_data['pmid']]['abstract']
    test_data['text'] = test_data['text'].lower().strip().replace('  ', ' ')
    test_data['text_tokens'] = m.encode(test_data['text'])
    test_data['text_tokens_with_prefix'] = torch.cat([test_data['text_tokens'], prefix], dim=-1).unsqueeze(0).cuda()
    test_data['pred_Ts'] = pred_d_not_in_original[id]['predT_not_in_original']
    test_data['pred_Ts_tokens'] = [m.encode(pred_T) for pred_T in test_data['pred_Ts']]
    test_data['gold_triples'] = gold_d[test_data['pmid']]['triples']
    return test_data

test_data = get_test_data(66)

print(f'{{\n"pred_Ds": "{test_data["pred_Ts"][0]}",')
print('"gold": {')
for key, value in test_data["gold_triples"][0].items():
    print(f'"{key}": "{value}",')
print('},')
# print(f'"gold": {test_data["gold_triples"][0]},')
print(f'"text": "{test_data["text"]}"\n}}')


# initialize
test_input = test_data['text_tokens_with_prefix']

output_text = []
k = 5
prob = []
ranking = []
step = 0

with torch.no_grad():
    for i in range(3):
        m.models[0].decoder.eval()
        step += 1

        out = m.models[0].decoder(test_input)

        softmax_out = torch.softmax(out[0][0][-1], dim=-1)
        _, top_k_indices = torch.topk(out[0][0][-1], k=k)
        top_k_tokens = [tokenizer.convert_ids_to_tokens([indice]) for indice in top_k_indices]
        top_k_probs = torch.softmax(out[0][0][-1][top_k_indices], dim=-1)
        top_k = [(token, prob.item()) for token, prob in zip(top_k_tokens, top_k_probs)]
        # print(f'The top-{k} most possible tokens are:\n{top_k}')
        next_token_id = 1
        test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
        output_text.append(top_k_indices[next_token_id-1])

        prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
        ranking.append(next_token_id)

    while (next_token_id != 0):        
        print(f'output_text: {m.decode(output_text)}\n')
        step += 1

        out = m.models[0].decoder(test_input)
        softmax_out = torch.softmax(out[0][0][-1], dim=-1)
        _, top_k_indices = torch.topk(out[0][0][-1], k=k)
        top_k_tokens = [tokenizer.convert_ids_to_tokens([indice]) for indice in top_k_indices]
        top_k_probs = torch.softmax(out[0][0][-1][top_k_indices], dim=-1)
        top_k = [(token, prob.item()) for token, prob in zip(top_k_tokens, top_k_probs)]
        print(f'The top-{k} most possible tokens are:\n{top_k}')
        next_token_id = int(input('Please input the next token (0 to end, -1 to input Customized string):'))
        if next_token_id == 0:
            break
        if next_token_id == -1:
            customized_string = input('Please input the Customized string:')
            customized_string_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(customized_string)).unsqueeze(0).cuda()
            test_input = torch.cat([test_input[0], customized_string_tokens], dim=-1).unsqueeze(0)
            output_text.append(customized_string_tokens.squeeze(0))

            customized_string_prob = out[0][0][-1][customized_string_tokens].clone()
            sorted_output, _ = torch.sort(out[0][0][-1], descending=True)
            print(f'The {customized_string} has the {torch.where(sorted_output == customized_string_prob)[0].item() + 1} / {len(sorted_output)} biggest probability in output.')
            
            prob.append(softmax_out[customized_string_tokens].item())
            ranking.append(torch.where(sorted_output == customized_string_prob)[0].item() + 1)

            continue
        if next_token_id != 1:
            sorted_output, _ = torch.sort(out[0][0][-1], descending=True)
            print(f'The token has the {next_token_id} / {len(sorted_output)} biggest probability in output.')
        test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
        output_text.append(top_k_indices[next_token_id-1])

        prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
        ranking.append(next_token_id)




import numpy as np
import matplotlib.pyplot as plt

# Create some fake data.
x = np.arange(step-1)
y1 = prob
y2 = ranking

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f'Pmid: {test_data["pmid"]}')

ax1.plot(x, y1, '.-')
ax1.set_ylabel('Probability')

ax2.plot(x, y2, '.-')
ax2.set_xlabel('step')
ax2.set_ylabel('Ranking')

marks = [0]* (step-1)
mark = False

# 0 for  parttern tokens, 1 for drugs ,2 for targets, 3 for interaction
for i, token in enumerate(output_text):
    if token != 6 and output_text[i-1] == 45:
        marks[i] = 1
        mark = True
        continue
    if token == 8 or token == 21 or token == 4:
        continue
    if token != 6 and output_text[i-1] == 8:
        marks[i] = 2
        mark = True
        continue
    if token != 6 and output_text[i-1] == 21:
        marks[i] = 3
        mark = True
        continue
    if token == 44:
        mark = False
        continue
    if mark:
        marks[i] = marks[i-1]
        

# if marks[x] == 1, then using hollow circle for the plot, if marks[x] == 2, then using hollow triangle for the plot, if marks[x] == 3, then using star for the plot.
for i in range(step-1):
    if marks[i] == 1:
        ax1.plot(x[i], y1[i], marker='o', color='white', markeredgecolor='blue')
        ax2.plot(x[i], y2[i], marker='o', color='white', markeredgecolor='blue')
        if y2[i] > 5:
            ax1.plot(x[i], y1[i], marker='o', color='white', markeredgecolor='red')
            ax2.plot(x[i], y2[i], marker='o', color='white', markeredgecolor='red')

    if marks[i] == 2:
        ax1.plot(x[i], y1[i], marker='^', color='white', markeredgecolor='blue')
        ax2.plot(x[i], y2[i], marker='^', color='white', markeredgecolor='blue')
        if y2[i] > 5:
            ax1.plot(x[i], y1[i], marker='^', color='red')
            ax2.plot(x[i], y2[i], marker='^', color='red')

    if marks[i] == 3:
        ax1.plot(x[i], y1[i], marker='x', color='white', markeredgecolor='blue')
        ax2.plot(x[i], y2[i], marker='x', color='white', markeredgecolor='blue')
        if y2[i] > 5:
            ax1.plot(x[i], y1[i], marker='x', color='red')
            ax2.plot(x[i], y2[i], marker='x', color='red')


plt.savefig(f'analysis/img/{test_data["pmid"]}.png')
plt.show()