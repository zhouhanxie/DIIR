import pandas as pd
from utils import (
OpenaiSequencialDialogue,
Session,
display_annomi_dialogue
)
from nltk.util import flatten
from sentence_transformers import SentenceTransformer
from tom_detector import determine_toms

from tom_detector import determine_toms_inference_mode
from utils import OPENAI_API_KEY
import openai
openai.api_key = OPENAI_API_KEY
from sentence_transformers import util
from utils import OpenaiSequencialDialogue
import numpy as np
from tqdm import tqdm


def rerank(state, client_tom, rules):
    reranker = OpenaiSequencialDialogue(model='gpt-3.5-turbo')
    reranker_prompt = """Look at the following description of a current situation (of a dialogue context), and a set of guidelines that might help in such situation. Response with the id of the single most applicable rule.
Dialogue Context:
@state@

Current Situation:
@client_tom@

Guidelines:
@rules@

Which rule applies the best in this situation? Answer with a single number only, do not explain anything, do not add punctuation."""
    reranker_prompt = reranker_prompt.replace('@state@', state).replace('@client_tom@', client_tom).replace('@rules@', rules)
    model_response = reranker.send_user_message(reranker_prompt)
    model_response = ''.join([str(i) for i in str(model_response) if i.isnumeric()])
    model_response = int(model_response)-1
    
    return model_response, reranker.cost()['total']

def inference_state(
    state,
    training_memory,
    client_tom_embeddings,
    sentence_encoder,
    verbose=True,
    receiver_lm_name = 'gpt-3.5-turbo-1106'
):

    def vprint(x): 
        if verbose:
            print('''---\n'''+x)
    client_stage, client_tom, determine_tom_cost = determine_toms_inference_mode(state)
    rules = list(training_memory['rule'])
    training_client_toms = list(training_memory['client tom'])
    
    # retrieve relevant training experience
    retriever_ranked_list = np.argsort(-util.dot_score(sentence_encoder.encode(client_tom), client_tom_embeddings).flatten(), axis=-1)
    top_rules = np.array(rules)[retriever_ranked_list[:10]]
    top_rules_text = '\n'.join([str(i+1)+': '+top_rules[i]+'\n' for i in range(len(top_rules))])
    # vprint('Top Retrieved Rules:\n'+top_rules_text)
    try:
        reranker_output, reranker_cost = rerank(state, client_tom, top_rules_text)
        # vprint('Reranker Selected Rule Number (starting from 1):\n'+str(reranker_output+1))
        retrieved_rule = top_rules[reranker_output]
        # vprint('Reranker Selected Rule:\n'+retrieved_rule)
    except Exception as e:
        # this is caused by not having enough relevant pieces in the training data
        # or the rerank model somehow returned something that cannot be parsed to a dictionary
        print('oops, something is off with the reranking step')
        print(e)
        retrieved_rule = top_rules[0]

    # setup the receiver
    receiver = OpenaiSequencialDialogue(model=receiver_lm_name, stop=['\n'])

    # inference
    vprint('State: '+state)
    vprint('Client Tom: '+client_tom)
    receiver_initial_prompt = f'Look at the following therapist-client dialogue, predict what should the therapist say next.\n{state}\n\nStart your resposne with [therapist]: '
    receiver_initial_response = receiver.send_user_message(receiver_initial_prompt)
    vprint('Receiver Initial Response: '+receiver_initial_response)
    vprint('Applying rule: ' + retrieved_rule)
    receiver_improvement_prompt = f'Look at the following therapist-client dialogue, predict what should the therapist say next.\n{state}.'+\
        f'\nFollow these guidelines when producing response:\n{retrieved_rule}\n\nStart your resposne with [therapist]: '
    receiver_updated_response = receiver.send_user_message(
                receiver_improvement_prompt
            )
    vprint('Receiver Updated Response: '+receiver_updated_response)
    receiver_cost = receiver.cost()['total']

    output = {
        'initial response':receiver_initial_response,
        'receiver response':receiver_updated_response,
        'client tom':client_tom,
        'retrieved_rule':retrieved_rule,
        'context': state,
        'cost breakdown': {'determine_tom_cost':determine_tom_cost, 'reranker_cost':reranker_cost, 'receiver_cost':receiver_cost},
        'cost': receiver_cost+determine_tom_cost+reranker_cost,
        'receiver_lm_name': receiver_lm_name
    }
    return output


def main(args):
    
    # load the data
    test_data = pd.read_csv(args.data_path)
    test_dialogue_ids = sorted(list(test_data['transcript_id'].unique()))

    sessions_to_play = []
    for i in test_dialogue_ids:
        sessions_to_play.append(Session(display_annomi_dialogue(test_data, i)))
        
    print('successfully built a bunch of sessions: ', len(sessions_to_play))
    num_dialogues_to_use = args.num_dialogues_to_use
    print(f'using {num_dialogues_to_use}')
    sessions_to_play = sessions_to_play[:num_dialogues_to_use]

    all_test_states = []
    all_test_gold_actions = []

    for session in sessions_to_play:
        session_states = []
        session_gold_actions = []
        
        session.state_tracker = 3
        session.skip_to_next_utt_by('therapist')
        try:
            while True:
                state, gold_action = session.state(), session.gold_action()
                if len(session.gold_action()) > 35:
                    session_states.append(state)
                    session_gold_actions.append(gold_action)
                session.skip_to_next_utt_by('therapist')
        except IndexError:
            all_test_states.append(session_states)
            all_test_gold_actions.append(session_gold_actions)

    all_test_states = flatten(all_test_states)
    all_test_gold_actions = flatten(all_test_gold_actions)
    print('num state and actions: ', len(all_test_states))


    # setup embeddings, training memory lookup, etc...
    training_memory = pd.read_csv(args.training_memory_path)
    from sentence_transformers import SentenceTransformer
    sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    client_tom_embeddings = sentence_encoder.encode(list(training_memory['client tom']))

    # start testing
    total_cost = 0
    num_state_action_pairs = len(all_test_states)
    play_outcomes = []
    for i in tqdm(list(range(num_state_action_pairs))):
        inference_output = inference_state(
            state = all_test_states[i],
            training_memory = training_memory,
            client_tom_embeddings = client_tom_embeddings,
            sentence_encoder = sentence_encoder,
            verbose=True,
            receiver_lm_name = args.receiver_lm_name
        )
        inference_output['gold action'] = all_test_gold_actions[i]
        play_outcomes.append(inference_output)
        total_cost += inference_output['cost']
        print(f'Total Cost So Far: {total_cost}')

    # saving
    pd.DataFrame(play_outcomes).to_csv(args.output_csv_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training the Agent')
    parser.add_argument('--data_path', type=str, default=None,
                        help='csv to the test data')
    parser.add_argument('--output_csv_file', type=str, default=None,
                        help='csv file to output the inference results')
    parser.add_argument('--receiver_lm_name', type=str, default='gpt-3.5-turbo-1106',
                        help='openai name of the receiver lm')
    parser.add_argument('--num_dialogues_to_use', type=int, default=5,
                        help='how many dialogue to use, if you have more than you want')
    parser.add_argument('--training_memory_path', type=str, default='',
                        help='path to the csv file generated by training')
    args = parser.parse_args()
    print(args)
    main(args)