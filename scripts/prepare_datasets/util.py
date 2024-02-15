import torch
import openai
import json
from tqdm import trange

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True,truncation=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens

def pooling(token_embeddings, attention_mask,weighted_mean=True):
    if weighted_mean:
        attention_mask *= attention_mask.cumsum(dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_llama2_embedding(model,tokenizer,texts,batch_size):
    print('calculate llama2 embedding')
    representation = []
    with torch.no_grad():
        for start_idx in trange(0,len(texts),batch_size):
            cur_texts = texts[start_idx:start_idx+batch_size]
            input_tokens = prepare_input(tokenizer=tokenizer, prompts=cur_texts)
            psg_out = model(**input_tokens, return_dict=True)
            p_reps = pooling(psg_out.last_hidden_state, input_tokens['attention_mask']).cpu()
            # p_reps = torch.nn.functional.normalize(p_reps, dim=-1).cpu()
            del psg_out
            representation.append(p_reps)
            # if representation is None:
            #     representation = p_reps
            # else:
            #     representation = torch.cat((representation,p_reps), 0)
    return torch.cat(representation, 0)

def worker(arg_dict):
    openai.api_key = 'YOUR_API_KEY'
    response = None
    prompt = f'''Text: {arg_dict['query']}

'''
    prompt += '''What is the domain of the text above?
Please reply with the json format: {"domain": ...}
'''
    ori_messages = [
                    {
                      "role": "system",
                      "content": "You are a good writer"
                    },
                    {
                      "role": "user",
                      "content": prompt
                    },
                  ]
    cur_messages = [
                    {
                      "role": "system",
                      "content": "You are a good writer"
                    },
                    {
                      "role": "user",
                      "content": prompt
                    },
                  ]
    exec_count = 0
    while response is None and exec_count<5:
        exec_count += 1
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    temperature=0,
                                                    request_timeout=10,
                                                    messages=cur_messages,
                                                    top_p=0)
            try:
                cur_dict = json.loads(response["choices"][0]["message"]["content"])
                assert 'domain' in cur_dict
                return {
                    'query': arg_dict['query'],
                    'domain': cur_dict['domain'],
                    'idx':arg_dict['idx']
                }
            except:
                print('invalid response',response["choices"][0]["message"]["content"].strip())
                cur_messages = ori_messages + [{
                    'role': 'assistant',
                    'content': response["choices"][0]["message"]["content"],

                }]
                response = None
        except Exception as e:
            print(e)

