# from transformers import BioGptTokenizer
import torch
import time
import os
# import spacy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def predict(m, n, pmids, gold_d, min_prob, tokenizer, nlp):
    tokenizer = tokenizer
    min_prob = min_prob
    nlp = nlp
    pmids = pmids
    gold_d = gold_d
    m = m
    n = n

    # process the input
    def get_test_data(id):
        prefix = torch.arange(42384, 42393)
        test_data = {}
        test_data['pmid'] = pmids[id]
        test_data['text'] = gold_d[test_data['pmid']]['title'].strip() + " " + gold_d[test_data['pmid']]['abstract']
        if type(test_data['text'][-1]) is int:
        # no too long text
            if len(test_data['text']) > 800:
                test_data['text'] = test_data['text'][:800]
            test_data['text_tokens'] = torch.Tensor(test_data['text']).long()

        # for tokenization optimized input file
        else:
                # no too long text
            if len(test_data['text']) > 4500:
                test_data['text'] = test_data['text'][:4500]
        # get the entity with NER
        original_entities = nlp(test_data['text'].strip().replace('  ', ' '))
        # get no-duplicate and lower entities, the list will revert the seq, to keep the seq of the entities reference the occur seq in the text [-1:]
        test_data['entities'] = list(set([entity.text.lower() for entity in original_entities.ents]))
        test_data['entities'] = test_data['entities'][::-1]
        l = len(test_data['entities'])
        # add the () for the short entities
        for i in range(l):
            if test_data['entities'][i][-1] == ')' and test_data['entities'][i][0] == '(':
                test_data['entities'][i] = test_data['entities'][i][1:-1]
            # print(f"entity : {test_data['entities'][i]}, length: {len(test_data['entities'][i])}")
            if len(test_data['entities'][i]) < 6:
                test_data['entities'].append(f"({test_data['entities'][i]})")
        test_data['entity_tokens'] = [m.encode(entity)[:-1] for entity in test_data['entities']]
        test_data['text'] = test_data['text'].lower().strip().replace('  ', ' ')
        # no "2" in the end
        test_data['text_tokens'] = m.encode(test_data['text'])[:-1]
        test_data['text_tokens_with_prefix'] = torch.cat([test_data['text_tokens'], prefix], dim=-1).unsqueeze(0).cuda()
        test_data['gold_triples'] = gold_d[test_data['pmid']]['triples']
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
                    if i == 0 and id not in self.interaction_dict["[0]"]:
                        self.interaction_dict["[0]"].append(id)
                    try:
                        if str(ids) in self.interaction_dict.keys() and token_ids[i+1] not in self.interaction_dict[str(ids)]:
                            self.interaction_dict[str(ids)].append(token_ids[i+1])
                        else:
                            self.interaction_dict[str(ids)] = [token_ids[i+1]]
                    except:
                        if str(ids) not in self.interaction_dict.keys():
                            self.interaction_dict[str(ids)] = None
    
    # predict the drug
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
            and_id=tokenizer.encode("and", add_special_tokens=False, return_tensors="pt").squeeze(0).squeeze(0),
            is_id=tokenizer.encode("is", add_special_tokens=False, return_tensors="pt").squeeze(0).squeeze(0),
            min_prob=min_prob):

        entity_tokens = entity_tokens
        and_id = and_id
        is_id = is_id
        min_prob=min_prob
        drugs = drugs
        targets = targets
        drug_target_pairs = drug_target_pairs

        # predict the drug
        torch.cuda.empty_cache()


        def next_entity_token(output_text, possible_entity_token, tokens, drug=False, target=False):
            interaction_token_ids = possible_entity_token
            
            if drug:
                # start_id = tokenizer.encode("between", add_special_tokens=False)[0]
                end_id = tokenizer.encode("and", add_special_tokens=False, return_tensors="pt").squeeze(0)
            else:
                # start_id = tokenizer.encode("and", add_special_tokens=False)[0]
                end_id = tokenizer.encode("is", add_special_tokens=False, return_tensors="pt").squeeze(0)

            # if it's the first output of interaction tokens
            if len(tokens) == 0:
                # print(f'next possible token with [0]: {interaction_token_ids.interaction_dict["[0]"]}')
                return [item.cuda()for item in interaction_token_ids.interaction_dict["[0]"]]
            
            else:
                current_output = [torch.tensor(token_id).cpu() for token_id in tokens]
                # print("current_output: ", current_output)
                assert str(current_output) in interaction_token_ids.interaction_dict.keys()
                if interaction_token_ids.interaction_dict[str(current_output)] is None:
                    return end_id.cuda()
                else:
                    # print(f"current_output: {current_output}, \nnext possible token: {interaction_token_ids.interaction_dict[str(current_output)]}")
                    return [item.cuda() for item in interaction_token_ids.interaction_dict[str(current_output)]]


        max_step = 500
        first_token = False
            
        max_num_entities = 5

            # drug 
        last_output_text = [x.clone() for x in output_text]
        last_test_input = test_input.clone()
        last_step = step
        # last_prob = prob
        # last_ranking = ranking

        possible_entity_token = interaction_tokens(test_data['entity_tokens'], file=False)

        choose_which_drug = 0
        need_to_predict_new_drug = False
        # number = [tokenizer.encode(str(num), add_special_tokens=False, return_tensors="pt").squeeze(0).squeeze(0).cuda() for num in range(10)]
        while(step < max_step):
            sub_entity = []
            drug_token_ids = []

            choose_which = 0
            possible_for_new_sub = False
            while(step < max_step):
                # print("current num_sub_entities: ", len(drug_token_ids))
                next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=True, target=False)
                # if there is only end_id for possible next token, then if the entities of span < 5, next possible token is and_id + all start tokens of all entities
                if and_id == next_token_in_entity:
                    # print("and_id == next_token_in_entity")
                    drug_token_ids.append(sub_entity)
                    sub_entity = []
                    # if new sub entity possible, next token list
                    if len(drug_token_ids) < max_num_entities:
                        possible_for_new_sub = True
                        # how_many_entities += 1
                        if sub_entity == []:
                            # print("having a new possible for sub_entity + and_id...")
                            next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=True, target=False)
                            next_token_in_entity.append(and_id.cuda())
                            # print(f"new possible for sub_entity + and_id\n{next_token_in_entity}")
                        else:
                            next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=True, target=False)
                            # print(f"new possible for sub_entity:\n{next_token_in_entity}")

                # print("possible next drug token:", next_token_in_entity)
                step += 1
                if len(next_token_in_entity) > 1:
                    choose_which = 0
                    # print("running the model")
                    out = m.models[0].decoder(test_input)
                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(softmax_out, len(softmax_out))

                    for i, token in enumerate(top_k_indices):
                        # choose_which_drug > 0 means to choose the second highest prob token
                        # but only the first token of the drug
                        # so here we set a flag to identify the first token of the drug: first_token
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
                    # print(f"the output_text after normal inference: \n{output_text}\n{m.decode(output_text)}")

                # only one choice in the next_token_in_entity    
                else:
                    # print("only one choice: ", next_token_in_entity, m.decode(next_token_in_entity))
                    next_token_id = next_token_in_entity[-1].cuda()
                    test_input = torch.cat([test_input[0], next_token_id.unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(next_token_id)
                    
                first_token = False

                # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                # ranking.append(next_token_id)
                if output_text[-1] == and_id:
                    possible_for_new_sub = False
                    new_drug = [id for drug in drug_token_ids for id in drug]
                    drugs.append(new_drug)
                    sub_entity = []
                        
                    break
                    # for sub-entities
                else:
                    sub_entity.append(output_text[-1])
                    # print("sub_entity for drug: ", m.decode(sub_entity))
                
                # print(f"the output_text before a new token: \n{m.decode(output_text)}")
                # print("next drug token...")
                
            # print(f"the output_text after drug: \n{output_text}\n{m.decode(output_text)}")
            # target
            # print("start for target...")
            target_token_ids = []
            no_target_output_text = [x.clone() for x in output_text]
            no_target_step = step
            no_target_input = test_input.clone()
            choose_which_target = 0
            sub_entity = []

            # a flag to identify the first token of the target, keep False
            # first_token = True

            # a flag to identify if there is need to predict a new drug instead of current drug
            need_to_predict_new_drug = False

            while(step < max_step):
                # target (single entity mode)
                next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=False, target=True)

                if is_id == next_token_in_entity:
                    target_token_ids.append(sub_entity)
                    
                    sub_entity = []
                    if len(target_token_ids) < max_num_entities:
                        possible_for_new_sub = True
                        if sub_entity == []:
                            # print("having a new possible for sub_entity + is_id...")
                            next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=False, target=True)
                            next_token_in_entity.append(is_id.cuda())
                            # print(f"new possible for sub_entity + is_id (target)\n{next_token_in_entity}")
                        else:
                            next_token_in_entity = next_entity_token(_, possible_entity_token, sub_entity, drug=False, target=True)
                            # print(f"new possible for sub_entity (target):\n{next_token_in_entity}")
                # print("possible next target token:", next_token_in_entity)
                step += 1
                # print("shape of test_input: ", test_input.shape)


                if len(next_token_in_entity) > 1:
                    choose_which = 0
                    out = m.models[0].decoder(test_input)
                    softmax_out = torch.softmax(out[0][0][-1], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(softmax_out, len(softmax_out))
                    for i, token in enumerate(top_k_indices):
                        # choose_which_target > 0 means to choose the second highest prob token
                        # but only the first token of the target
                        # so here we set a flag to identify the first token of the target
                        if token in next_token_in_entity:
                            if not first_token:
                                next_token_id = i + 1
                                break
                            else:
                                if choose_which < choose_which_target:
                                    choose_which += 1
                                    if top_k_probs[i].item() >= min_prob:
                                        next_token_id = i + 1
                                    else:
                                        print(f"no more possible targets for {m.decode(drugs[-1])}, try to predict a new drug...")
                                        need_to_predict_new_drug = True
                                    break

                    # if there is no possible target for current drug, then we need to predict a new drug, stop the while loop for target and go to continue
                    if need_to_predict_new_drug:
                        break


                    test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(top_k_indices[next_token_id-1])
                else:
                    # print("only one choice (target): ", next_token_in_entity, m.decode(next_token_in_entity))
                    next_token_id = next_token_in_entity[-1].cuda()
                    test_input = torch.cat([test_input[0], next_token_id.unsqueeze(0)], dim=-1).unsqueeze(0)
                    output_text.append(next_token_id)

                first_token = False

                # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                # ranking.append(next_token_id)
                if output_text[-1] == is_id:
                    # print("output_text[-1] == is_id")
                    possible_for_new_sub = False
                    new_target = [id for target in target_token_ids for id in target]
                    pair = new_drug + new_target
                    # print("pair: \n", pair)
                    # print("drug_target_pairs: \n", drug_target_pairs)
                    sub_entity = []
                    if pair not in drug_target_pairs:
                        # print("keep going")
                        # print(f'with find target:\n {m.decode(output_text)}\n')
                        targets.append(new_target)
                        drug_target_pairs.append(pair)

                        # print("the final output w target is: \n", m.decode(output_text))
                        return output_text, test_input, step, drug_target_pairs#, prob, ranking
                    
                    # a new target
                    else:
                        output_text = [x.clone() for x in no_target_output_text]
                        target_token_ids = []
                        sub_entity = []
                        # step = no_target_step
                        test_input = no_target_input.clone()
                        # prob = no_target_prob
                        # ranking = last_ranking
                        choose_which_target += 1
                        if choose_which_target > 5:
                            need_to_predict_new_drug = True
                            break
                        else:
                            print(f"duplicate target, try to predict {choose_which_target} target...")
                            # print(f"new input: {m.decode(output_text)}")
                            first_token = True

                else:
                    sub_entity.append(output_text[-1])
                    # print("sub_entity for target: ", m.decode(sub_entity))
                
                # print(f"the output_text before a new target token: \n{m.decode(output_text)}")
                # print("next target token...")

            if need_to_predict_new_drug:
                print("need_to_predict_new_drug")
                output_text = [x.clone() for x in last_output_text]
                test_input = last_test_input.clone()
                # step = last_step
                # prob = last_prob
                # ranking = last_ranking
                choose_which_drug += 1
                if choose_which_drug > 5:
                    print("no more possible drugs, the process will stop here.")
                return None
                need_to_predict_new_drug = False
                continue
            else:
                print("something wrong, stop here.")
                # break
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
            k=8000
    ): 
        def next_interaction_token(output_text, inter_token_ids, is_id=is_id):
            interaction_token_ids = interaction_tokens()
            next_or_end_id = tokenizer.encode("; .", add_special_tokens=False)

            # if it's the first output of interaction tokens
            if len(inter_token_ids) < 1:
                return interaction_token_ids.interaction_dict["[0]"]
            
            else:

                current_output = [token_id.item() for token_id in inter_token_ids]
                assert str(current_output) in interaction_token_ids.interaction_dict.keys()
                if interaction_token_ids.interaction_dict[str(current_output)] is None:
                    return next_or_end_id
                else:
                    return interaction_token_ids.interaction_dict[str(current_output)]
                
        inter_token_ids = []
        while(step < max_step):
            # print(f'output_text: {m.decode(output_text)}\n')
            
            next_word_in_original = next_interaction_token(output_text, inter_token_ids)
            # print("possible next interaction token:", next_word_in_original)
            step += 1
            # print("test_input: ", test_input)
            # print("length of test_input: ", len(test_input[0]))
            out = m.models[0].decoder(test_input)
            # print("out, ok")
            softmax_out = torch.softmax(out[0][0][-1], dim=-1)
            # print("softmax, ok")
            _, top_k_indices = torch.topk(out[0][0][-1], k=k)
            # print("top-k, ok")
            for i, token in enumerate(top_k_indices):
                # print("top-k token:", i, token.item())
                if token.item() in next_word_in_original:
                    next_token_id = i + 1
                    break
            test_input = torch.cat([test_input[0], top_k_indices[next_token_id-1].unsqueeze(0)], dim=-1).unsqueeze(0)
            output_text.append(top_k_indices[next_token_id-1])
            inter_token_ids.append(top_k_indices[next_token_id-1].unsqueeze(0))
            # print(f'inter_token_ids: {inter_token_ids}\n')

            # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
            # ranking.append(next_token_id)
            if output_text[-1] in [next_id, end_id]:
                # print(f'output_text: {m.decode(output_text)}\n')
                break

        return output_text, test_input, step#, prob, ranking


    # initialize
    start_time = time.time()
    # initialize
    test_data = get_test_data(n)
    # test_input_original = test_data['text_tokens_with_prefix']
    test_input = test_data['text_tokens_with_prefix']
    if len(test_data['entity_tokens']) < 1:
        
        print("no entities, skip this article")
        return [6, 639, 45]
    # test_text = test_data['text_tokens'].cuda()

    k = 8000

    print(f"start processing: {n + 1}/{len(pmids)}...")
    is_id = tokenizer.encode("is", add_special_tokens=False)[0]
    next_id = tokenizer.encode(";", add_special_tokens=False)[0]
    end_id = tokenizer.encode(".", add_special_tokens=False)[0]
    output_text = []
    prob = []
    ranking = []
    step = 0
    max_step = 500
    # num_tuples = 0
    drugs = []
    targets = []
    num_triplets = 0
    drug_target_pairs = []
    
    torch.cuda.empty_cache()
    with torch.no_grad():

        m.models[0].decoder.eval()

        while(step < max_step):
            # # there is an unsolved bug for n = 706
            # if n == 706:
            #     output_text = [6, 639,45]
            #     break

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

                # prob.append(softmax_out[top_k_indices[next_token_id-1]].item())
                # ranking.append(next_token_id)

            result = drug_target_prediction(
                output_text=output_text,
                test_input=test_input,
                drugs=drugs,
                targets=targets,
                drug_target_pairs=drug_target_pairs,
                step=step, 
                max_step=max_step,
                prob=prob,
                ranking=ranking,
                m=m,
                tokenizer=tokenizer,
                entity_tokens=test_data["entity_tokens"],
                min_prob=min_prob,
                )
            if result:
                output_text, test_input, step, drug_target_pairs = result
            else:
                if len(last_output_text) > 0:
                    output_text = [x.clone() for x in last_output_text]
                else:
                    output_text = [6, 639,45]
                # print(f'The final output:{m.decode(output_text)}\n')
                break

            # print(f'output_text: {m.decode(output_text)}\n')

            # print("test_input after d&t: ", test_input)
            # interactions
            output_text, test_input, step = interaction_prediction(output_text, test_input, step,
                                                                    max_step, prob, ranking, m, tokenizer, is_id, next_id, end_id, k)
            
            num_triplets += 1
            if output_text[-1] == end_id or step >= max_step or num_triplets >= 3:
                if step >= max_step:
                    print("Steps exceed the maximum steps")
                break
    end_time = time.time()
    print("final outputs: ", m.decode(output_text))
    print(f"{n + 1} / {len(pmids)}\nElapsed time: {end_time - start_time}s")

    return output_text


    
