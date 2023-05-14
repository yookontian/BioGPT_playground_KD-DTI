# from transformers import BioGptTokenizer
import torch
import time
# import spacy



def predict(m, pmids, gold_d, min_prob, tokenizer, nlp):
    tokenizer = tokenizer
    min_prob = min_prob
    # process the input

    def get_test_data(id):
        prefix = torch.arange(42384, 42393)
        test_data = {}
        test_data['pmid'] = pmids[id]
        test_data['text'] = gold_d[test_data['pmid']]['title'].strip() + " " + gold_d[test_data['pmid']]['abstract']
    
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
        # get the entity with NER
        original_entities = nlp(test_data['text'].strip().replace('  ', ' '))
        # get no-duplicate and lower entities
        test_data['entities'] = list(set([entity.text.lower() for entity in original_entities.ents]))
        test_data['entity_tokens'] = [m.encode(entity)[:-1] for entity in test_data['entities']]
        test_data['text'] = test_data['text'].lower().strip().replace('  ', ' ')
        # no "2" in the end
        test_data['text_tokens'] = m.encode(test_data['text'])[:-1]
        test_data['text_tokens_with_prefix'] = torch.cat([test_data['text_tokens'], prefix], dim=-1).unsqueeze(0).cuda()
        return test_data


    # process the interaction list for usage
    class interaction_tokens():
        def __init__(self, interactions_file='data/KD-DTI/raw/interactions', file=True):
            if file:
                interactions = []
                with open (interactions_file, 'r') as f:
                    for line in f:
                        line = line.rstrip()
                        interactions.append(line)
                interaction_tokens = [tokenizer.encode(interaction, add_special_tokens=False) for interaction in interactions]
            else:
                interaction_tokens = interactions_file

            
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

    # predict the drug and targry
    def drug_target_prediction(
            output_text,
            test_input,
            drugs,
            targets,
            drug_target_pairs,
            step, 
            max_step,
            prob,
            ranking,
            m,
            tokenizer,
            entity_tokens,
            min_prob,
            between_id=tokenizer.encode("between", add_special_tokens=False)[0],
            and_id=tokenizer.encode("and", add_special_tokens=False)[0],
            is_id=tokenizer.encode("is", add_special_tokens=False)[0],
            ):

        def next_entity_token(output_text, possible_entity_token, drug=False, target=False):
            interaction_token_ids = possible_entity_token
            
            if drug or (not target):
                start_id = tokenizer.encode("between", add_special_tokens=False)[0]
                end_id = tokenizer.encode("and", add_special_tokens=False, return_tensors="pt").squeeze(0)
            else:
                start_id = tokenizer.encode("and", add_special_tokens=False)[0]
                end_id = tokenizer.encode("is", add_special_tokens=False, return_tensors="pt").squeeze(0)

            # if it's the first output of interaction tokens
            if output_text[-1] == start_id:
                return interaction_token_ids.interaction_dict["[0]"]
            
            else:

                # get the current latest outputted interaction tokens
                for i in range(len(output_text)-1, 0, -1):
                    if output_text[i] == start_id:
                        current_output = output_text[i+1:]
                        break
                current_output = [token_id.cpu() for token_id in current_output]
                # print("current_output: ", current_output)
                assert str(current_output) in interaction_token_ids.interaction_dict.keys()
                if interaction_token_ids.interaction_dict[str(current_output)] is None:
                    return end_id
                else:
                    return interaction_token_ids.interaction_dict[str(current_output)]

        
        # drug 
        last_output_text = [x.clone() for x in output_text]
        last_test_input = test_input.clone()
        last_step = step
        last_prob = prob
        last_ranking = ranking

        possible_entity_token = interaction_tokens(entity_tokens, file=False)

        choose_which_drug = 0
        need_to_predict_new_drug = False

        while(step < max_step):
            drug_token_ids = []
            first_token = True
            choose_which = 0
            while(step < max_step):
                # drug (single entity mode)
                next_token_in_entity = next_entity_token(output_text, possible_entity_token, drug=True)
                # print("possible next drug token:", next_token_in_entity)
                step += 1
                out = m.models[0].decoder(test_input)
                softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                top_k_probs, top_k_indices = torch.topk(softmax_out, len(softmax_out))
                if len(next_token_in_entity) > 1:
                    choose_which = 0
                    for i, token in enumerate(top_k_indices):
                        # choose_which_drug > 0 means to choose the second highest prob token
                        # but only the first token of the drug
                        # so here we set a flag to identify the first token of the drug: first_token
                        if token in next_token_in_entity:
                            if token in next_token_in_entity:
                                if not first_token:
                                    next_token_id = i + 1
                                    break
                                else:
                                    if choose_which < choose_which_drug:
                                        choose_which += 1
                                    else:
                                        if top_k_probs[i].item() >= min_prob:
                                            next_token_id = i + 1
                                        else:
                                            print(f"no more possible drugs, the process will stop here.")
                                            return None
                                        break
                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])
                    drug_token_ids.append(top_k_indices[next_token_id-1].item())
                else:
                    # print("only one choice: ", next_token_in_entity)
                    next_token_id = next_token_in_entity[-1].cuda()
                    test_input = torch.cat([test_input[0], next_token_id.unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(next_token_id)
                    drug_token_ids.append(next_token_id.item())

                first_token = False

                # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                # ranking.append(next_token_id)
                if output_text[-1] == and_id:
                    # print(f'drug: {m.decode(output_text)}\n')
                    drugs.append(drug_token_ids[:-1])
                    break
                

            # target
            target_token_ids = []
            no_target_output_text = [x.clone() for x in output_text]
            no_target_step = step
            no_target_input = test_input.clone()
            choose_which_target = 0

            # a flag to identify the first token of the target
            first_token = True

            # a flag to identify if there is need to predict a new drug instead of current drug
            need_to_predict_new_drug = False

            while(step < max_step):
                # target (single entity mode)
                next_token_in_entity = next_entity_token(output_text, possible_entity_token, target=True)
                # print("possible next target token:", next_token_in_entity)
                step += 1
                out = m.models[0].decoder(test_input)
                softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                top_k_probs, top_k_indices = torch.topk(softmax_out, len(softmax_out))
                if len(next_token_in_entity) > 1:
                    choose_which = 0
                    for i, token in enumerate(top_k_indices):
                        # choose_which_target > 0 means to choose the second highest prob token
                        # but only the first token of the target
                        # so here we set a flag to identify the first token of the target
                        if token in next_token_in_entity:
                            if token in next_token_in_entity:
                                if not first_token:
                                    next_token_id = i + 1
                                    break
                                else:
                                    if len(drug_target_pairs) < 1:
                                        next_token_id = i + 1
                                    else:
                                        if choose_which < choose_which_target:
                                            choose_which += 1
                                        else:
                                            if top_k_probs[i].item() >= min_prob:
                                                next_token_id = i + 1
                                            else:
                                                # print(f"no more possible targets for {m.decode(drugs[-1])}, try to predict a new drug...")
                                                need_to_predict_new_drug = True
                                        break
                    # if there is no possible target for current drug, then we need to predict a new drug, stop the while loop for target and go to continue
                    if need_to_predict_new_drug:
                        break


                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])
                    target_token_ids.append(top_k_indices[next_token_id-1].item())
                else:
                    # print("only one choice: ", next_token_in_entity)
                    next_token_id = next_token_in_entity[-1].cuda()
                    test_input = torch.cat([test_input[0], next_token_id.unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(next_token_id)
                    target_token_ids.append(next_token_id.item())

                first_token = False

                # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                # ranking.append(next_token_id)
                if output_text[-1] == is_id:
                    pair = drug_token_ids[:-1] + target_token_ids[:-1]
                    if pair not in drug_target_pairs:
                        # print(f'target: {m.decode(output_text)}\n')
                        targets.append(target_token_ids[:-1])
                        drug_target_pairs.append(pair)

                        return output_text, test_input, step, drug_target_pairs#, prob, ranking
                    
                    else:
                        output_text = no_target_output_text
                        step = no_target_step
                        test_input = no_target_input
                        # prob = no_target_prob
                        # ranking = last_ranking
                        choose_which_target += 1
                        # print(f"duplicate target, try to predict {choose_which_target} target...")
                        first_token = True
                        target_token_ids = []


            if need_to_predict_new_drug:
                output_text = [x.clone() for x in last_output_text]
                test_input = last_test_input.clone()
                step = last_step
                prob = last_prob
                ranking = last_ranking
                choose_which_drug += 1
                need_to_predict_new_drug = False
                continue
            else:
                print("something wrong, stop here.")
                return None

    # predict interaction
    def interaction_prediction(
            output_text,
            test_input,
            step, 
            max_step,
            prob,
            ranking,
            m,
            tokenizer,
            is_id,
            next_id,
            end_id,
            k
    ): 
        def next_interaction_token(output_text, is_id=is_id):
            interaction_token_ids = interaction_tokens()
            next_or_end_id = tokenizer.encode("; .", add_special_tokens=False)

            # if it's the first output of interaction tokens
            if output_text[-1] == is_id:
                return interaction_token_ids.interaction_dict["[0]"]
            
            else:

                # get the current latest outputted interaction tokens
                for i in range(len(output_text)-1, 0, -1):
                    if output_text[i] == is_id:
                        current_output = output_text[i+1:]
                        break
                current_output = [token_id.item() for token_id in current_output]
                assert str(current_output) in interaction_token_ids.interaction_dict.keys()
                if interaction_token_ids.interaction_dict[str(current_output)] is None:
                    return next_or_end_id
                else:
                    return interaction_token_ids.interaction_dict[str(current_output)]
                

        while(step < max_step):
            # print(f'output_text: {m.decode(output_text)}\n')
            
            next_word_in_original = next_interaction_token(output_text)
            # print("possible next interaction token:", next_word_in_original)
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
                # print(f'output_text: {m.decode(output_text)}\n')
                break

        
        return output_text, test_input, step#, prob, ranking
    



    outputs = []
    for n, id in enumerate(pmids):
        # initialize
        start_time = time.time()
        test_data = get_test_data(n)
        test_input = test_data['text_tokens_with_prefix']

        k = 8000

        is_id = tokenizer.encode("is", add_special_tokens=False)[0]
        next_id = tokenizer.encode(";", add_special_tokens=False)[0]
        end_id = tokenizer.encode(".", add_special_tokens=False)[0]
        output_text = []
        prob = []
        ranking = []
        step = 0
        max_step = 500
        drugs = []
        targets = []
        drug_target_pairs = []

        with torch.no_grad():
            m.models[0].decoder.eval()
            while(step < max_step):
                last_output_text = [x.clone() for x in output_text]
                last_prob = prob
                last_ranking = ranking
                last_step = step
                for i in range(3):
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

                result = drug_target_prediction(
                    output_text,
                    test_input,
                    drugs,
                    targets,
                    drug_target_pairs,
                    step, 
                    max_step,
                    prob,
                    ranking,
                    m,
                    tokenizer,
                    test_data["entity_tokens"],
                    min_prob
                    )
                if result:
                    output_text, test_input, step, drug_target_pairs = result
                else:
                    output_text = last_output_text
                    # print(f'The final output:{m.decode(output_text)}\n')
                    break

                # print(f'output_text: {m.decode(output_text)}\n')

                # interactions
                output_text, test_input, step = interaction_prediction(output_text, test_input, step,
                                                                        max_step, prob, ranking, m, tokenizer, is_id, next_id, end_id, k)
                
                if output_text[-1] == end_id or step >= max_step:
                    # if step >= max_step:
                        # print("Steps exceed the maximum steps")
                    break
            outputs.append(output_text)
            end_time = time.time()
            print(f"{len(outputs)} / {len(pmids)}\nElapsed time: {end_time - start_time}s")



    return outputs
