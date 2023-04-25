from transformers import BioGptTokenizer
import torch



def predict(m, inputs):
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    # process the input
    def get_test_data(input):
        prefix = torch.arange(42384, 42393)
        test_data = {}
        test_data['text'] = input

        # for original input file
        if type(test_data['text'][-1]) is int:
            # no too long text
            if len(test_data['text']) > 850:
                test_data['text'] = test_data['text'][:850]
            test_data['text_tokens'] = torch.Tensor(test_data['text']).long()
        
        # for tokenization optimized input file
        else:
             # no too long text
            if len(test_data['text']) > 4800:
                test_data['text'] = test_data['text'][:4800]
            # no "2" in the end
            test_data['text_tokens'] = m.encode(test_data['text'])[:-1]
            
        test_data['text_tokens_with_prefix'] = torch.cat([test_data['text_tokens'], prefix], dim=-1).unsqueeze(0).cuda()
        return test_data
    


    # process the interaction list for usage
    class interaction_tokens():
        def __init__(self, interactions):
            interaction_tokens = [tokenizer.encode(interaction, add_special_tokens=False) for interaction in interactions]
            self.interaction_dict = {"[0]":[]}
            for token_ids in interaction_tokens:
                ids = []
                for i, id in enumerate(token_ids):
                    ids += [id]
                    if i == 0:
                        self.interaction_dict["[0]"].append(id)
                    try:
                        if str(ids) in self.interaction_dict.keys():
                            self.interaction_dict[str(ids)].append(token_ids[i+1])
                        else:
                            self.interaction_dict[str(ids)] = [token_ids[i+1]]
                    except:
                        if str(ids) not in self.interaction_dict.keys():
                            self.interaction_dict[str(ids)] = None
            self.interaction_dict["[0]"] = list(set(self.interaction_dict["[0]"]))


    # have the interaction list
    interactions = []
    with open ('/home/tian/Projects/BioGPT/data/KD-DTI/raw/interactions', 'r') as f:
        for line in f:
            line = line.rstrip()
            interactions.append(line)
    interaction_tokens = interaction_tokens(interactions)

    # make a function for next possible token of interaction
    # pred_inter = predicted interaction tokens(if exists)
    def next_interaction_token(output_text, interaction_tokens=interaction_tokens):
        is_id = tokenizer.encode("is", add_special_tokens=False)[0]
        next_or_end_id = tokenizer.encode("; .", add_special_tokens=False)

        # if it's the first output of interaction tokens
        if output_text[-1] == is_id:
            return interaction_tokens.interaction_dict["[0]"]
        
        else:

            # get the current latest outputted interaction tokens
            for i in range(len(output_text)-1, 0, -1):
                if output_text[i] == is_id:
                    current_output = output_text[i+1:]
                    break
            current_output = [token_id.item() for token_id in current_output]
            assert str(current_output) in interaction_tokens.interaction_dict.keys()
            if interaction_tokens.interaction_dict[str(current_output)] is None:
                return next_or_end_id
            else:
                return interaction_tokens.interaction_dict[str(current_output)]
            

    outputs = []
    with torch.no_grad():
        m.models[0].decoder.eval()

        k = 8000

        and_id = tokenizer.encode("and", add_special_tokens=False)[0]
        is_id = tokenizer.encode("is", add_special_tokens=False)[0]
        next_id = tokenizer.encode(";", add_special_tokens=False)[0]
        end_id = tokenizer.encode(".", add_special_tokens=False)[0]
        for n, input in enumerate(inputs):
                # initialize
            test_data = get_test_data(input)
            test_input = test_data['text_tokens_with_prefix'].cuda()
            test_text = test_data['text_tokens'].cuda()


            output_text = []
            prob = []
            ranking = []
            step = 0

            while(step < 150):
                # the interaction between
                for i in range(3):
                    if step > 150:
                        break
                    step += 1

                    out = m.models[0].decoder(test_input)

                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    _, top_k_indices = torch.topk(out[0][0][-1], k=1)
                    top_k_tokens = [tokenizer.convert_ids_to_tokens([indice]) for indice in top_k_indices]
                    top_k_probs = torch.softmax(out[0][0][-1][top_k_indices], dim=-1)
                    top_k = [(token, prob.item()) for token, prob in zip(top_k_tokens, top_k_probs)]
                    # print(f'The top-{k} most possible tokens are:\n{top_k}')
                    next_token_id = 1
                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])

                    prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                    ranking.append(next_token_id)

                # print(f'output_text: {m.decode(output_text)}\n')
                

                # drug
                # set drug = 0 to prevent the first token is "and"
                drug = 0
                while(step < 150):
                    # print(f'output_text: {m.decode(output_text)}\n')
                    if step > 150:
                        break
                    step += 1

                    out = m.models[0].decoder(test_input)
                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    _, top_k_indices = torch.topk(out[0][0][-1], k=k)
                    if drug == 0:
                        last_token = output_text[-1]
                        for i, token in enumerate(top_k_indices):
                            if token in test_text:
                                next_token_id = i + 1
                                drug = 1
                                break
                    else:
                        next_word_in_original = []
                        try:
                            for i, token_id in enumerate(test_text):
                                if token_id  == output_text[-1] and output_text[-2] == last_token:
                                    next_word_in_original.append(test_text[i + 1])
                                else:
                                    # set the window_size = 2
                                    if token_id  == output_text[-1] and output_text[-2] == test_text[i - 1]:
                                        next_word_in_original.append(test_text[i + 1])
                        except:
                            pass
                        for i, token in enumerate(top_k_indices):
                            if (token in next_word_in_original) or (token == and_id):
                                next_token_id = i + 1
                                break
                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])

                    # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                    # ranking.append(next_token_id)
                # and
                    if output_text[-1] == and_id and tokenizer.decode(output_text[-2]).endswith("</w>"):
                        break
                
                # print(f'output_text: {m.decode(output_text)}\n')

                # target
                target = 0
                while(step < 150):
                    # print(f'output_text: {m.decode(output_text)}\n')

                    step += 1

                    out = m.models[0].decoder(test_input)
                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    _, top_k_indices = torch.topk(out[0][0][-1], k=k)
                    if target == 0:
                        last_token = output_text[-1]
                        for i, token in enumerate(top_k_indices):
                            if token in test_text:
                                next_token_id = i + 1
                                target = 1
                                break
                    else:
                        next_word_in_original = []
                        try:
                            for i, token_id in enumerate(test_text):
                                if token_id  == output_text[-1] and output_text[-2] == last_token:
                                    next_word_in_original.append(test_text[i + 1])
                                else:
                                    # set the window_size = 2
                                    if token_id  == output_text[-1] and output_text[-2] == test_text[i - 1]:
                                        next_word_in_original.append(test_text[i + 1])
                        except:
                            pass
                        for i, token in enumerate(top_k_indices):
                            if (token in next_word_in_original) or (token == is_id):
                                next_token_id = i + 1
                                break
                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])

                    # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                    # ranking.append(next_token_id)
                # is
                    if output_text[-1] == is_id and tokenizer.decode(output_text[-2]).endswith("</w>"):
                        break

                # interaction and next_or_end token
                while(step < 150):
                    # print(f'output_text: {m.decode(output_text)}\n')

                    if step > 150:
                        break

                    next_word_in_original = next_interaction_token(output_text)


                    step += 1
                    out = m.models[0].decoder(test_input)
                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    _, top_k_indices = torch.topk(out[0][0][-1], k=k)
                    for i, token in enumerate(top_k_indices):
                        if token in next_word_in_original:
                            next_token_id = i + 1
                            break
                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])

                    # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                    # ranking.append(next_token_id)
                    if output_text[-1] in [next_id, end_id]:
                        break

                if output_text[-1] == end_id or step >= 150:
                    # print(f'final output_text: {m.decode(output_text)}\n')
                    outputs.append(output_text)
                    print(f"{len(outputs)} / {len(inputs)}\n")

                    break


    
    return outputs