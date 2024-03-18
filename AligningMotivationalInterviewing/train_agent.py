import pandas as pd
from utils import (
OpenaiSequencialDialogue,
Session,
display_annomi_dialogue
)
from nltk.util import flatten
from utils import DialogueActClassifier 
import openai
from utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from tom_detector import determine_toms
from collections import defaultdict
from tqdm import tqdm


def truncate_labels(labels):
    """
    Filter rare MI-insonsistent labels from gold
    to void degradating model behavior.
    """
    out = []
    for l in labels:
        if l in ('Confront', 'Direct', 'Warn', 'Other'):
            pass
        elif 'reflection' in l.lower():
            out.append('Reflection')
        elif 'advise' in l.lower():
            out.append('Advise')
        elif 'question' in l.lower():
            out.append('Question')
        else:
            out.append(l)
        
    return out

def play_state(
    state,
    gold_action,
    dialog_act_classifier,
    verbose=True,
    max_align_loop = 3,
    sender_lm_name = 'gpt-3.5-turbo-1106',
    receiver_lm_name = 'gpt-3.5-turbo-1106'
):
    """
    play out a state-action pair
    """

    def vprint(x): 
        if verbose:
            print('''---\n'''+x)
    
    vprint('State:\n'+state)
    vprint('Gold Action:\n'+gold_action)

    # setup a marker to track whenever cost incurred
    tot_cost = defaultdict(int)
    data = dict()
    data['alignment successful'] = False
    
    # some analysis on state and action
    client_stage, client_tom, therapist_tom, contextual_instruction, cost = determine_toms(state, gold_action)
    tot_cost['tom inference'] += cost
    data['client stage'] = client_stage
    data['client tom'] = client_tom
    data['therapist tom'] = therapist_tom
    data['contextual instruction'] = contextual_instruction
    gold_sentences, gold_mi_codes, cost = dialog_act_classifier.annotate_dialogue_turn(state, gold_action)
    tot_cost['annotate gold actions'] += cost
    data['gold mi code'] = gold_mi_codes
    gold_mi_codes = truncate_labels(gold_mi_codes)
    vprint('Client TOM: '+client_tom)
    vprint('Annotated gold actions: '+str(gold_mi_codes))
    
    receiver = OpenaiSequencialDialogue(model=receiver_lm_name, stop=['\n'])
    sender = OpenaiSequencialDialogue(model=sender_lm_name)
    
    receiver_initial_prompt = f'Look at the following therapist-client dialogue, predict what should the therapist say next.\n{state}\n\nStart your resposne with [therapist]: '
    receiver_initial_response = receiver.send_user_message(receiver_initial_prompt)
    receiver.rewind()
    data['initial response'] = receiver_initial_response
    vprint('Initial response:\n'+receiver_initial_response)
    
    
    # do initialization of sender/receiver
    sender_prompt = 'You are trying to teach a student to follow true therapist\'s motivaional interviewing behavior. '+\
    'Here is the current scenario:\n'+\
    state+\
    '\n\nClient mental state seems to be: '+client_tom+\
    '\n\nThe Student Response is: \n'+receiver_initial_response+\
    '\nIn comparison, the true therapist response is: \n'+gold_action+\
    '\n\nFrom our annotation, it seems like the true therapist\'s actions in order, sentence by sentence, are: '+str(gold_mi_codes)+', which is '+str(len(gold_sentences))+' sentences.'\
    ' Analyze the current situation, and write a instruction for the student, in the format of '+\
    'Based on the annotation, When the client ..., the therapist should ..., the therapist should not ...'+\
    '\nWhen mention what the therapist should do, be sure to include information on how many sentence are needed and what each sentence should do.'+\
    '\nImportant: this is not a general guideline, but should be specifically tailored to the flaw in the student response.'+\
    'Be general, make sure your rule is generalizable across topics. For example, simple use \'bad habit\' instead of drug abuse/alcohol issue/smoking.'+\
    ' such that we can reuse this rule for other topics in the future.'
    
    content_rule = sender.send_user_message(sender_prompt).replace('Based on the annotation, ', '')
    vprint('Sender:\n'+content_rule)
    data['rule'] = content_rule
    receiver_improvement_prompt = f'Look at the following therapist-client dialogue, predict what should the therapist say next.\n{state}.'+\
    f'\nFollow these guidelines when producing response:\n{content_rule}\n\nStart your resposne with [therapist]: '
    receiver_updated_response = receiver.send_user_message(
                receiver_improvement_prompt
            )
    receiver.rewind()
    vprint('Receiver:\n'+receiver_updated_response)
    
    # looping through alignment
    success_check = ''
    loop = 0
    for i in range(max_align_loop):
        # check end condition
        success_check = sender.send_user_message(
            f'The student wrote a response based on your rule. {receiver_updated_response}, did the student correctly follow your guideline and replicated the true therapist? Answer yes or no first.'
        )
        vprint(f'success check: {success_check}')
        if 'yes' in success_check.lower():
            data['alignment successful'] = True
            break 
        loop += 1
        if loop >= max_align_loop:
            vprint(f'breaking alignment loop since max loop of {max_align_loop} reached')
    
        # else, improve
        content_rule = sender.send_user_message('Then update your instruction to better guide the student.').replace('Based on the annotation, ', '')
        data['rule'] = content_rule
        vprint('Sender:\n'+content_rule)
        receiver_improvement_prompt = f'Look at the following therapist-client dialogue, predict what should the therapist say next.\n{state}.'+\
        f'\nFollow these guidelines when producing response:\n{content_rule}\n\nStart your resposne with [therapist]: '
        receiver_updated_response = receiver.send_user_message(
                    receiver_improvement_prompt
                )
        receiver.rewind()
        vprint('Receiver:\n'+receiver_updated_response)


    data['num alignment loop'] = loop
    tot_cost['sender cost'] += sender.cost()['total']
    tot_cost['receiver cost'] += receiver.cost()['total']
    data['cost'] = sum(tot_cost.values())
    data['cost breakdown'] = dict(tot_cost)
    data['receiver response'] = receiver_updated_response
    data['context'] = state
    data['gold response'] = gold_action
    data['sender lm name'] = sender_lm_name
    data['receiver lm name'] = receiver_lm_name

    return data

def main(args):

    # building training data
    training_data = pd.read_csv(args.data_path)
    training_dialogue_ids = list(pd.read_csv(args.index_dir)['id'])

    sessions_to_play = []
    for i in training_dialogue_ids:
        sessions_to_play.append(Session(display_annomi_dialogue(training_data, i)))
        
    print('successfully built a bunch of sessions: ', len(sessions_to_play))
    print(f'using {args.num_dialogues_to_use}')
    sessions_to_play = sessions_to_play[args.begin_dialogue_position:args.begin_dialogue_position+args.num_dialogues_to_use]

    all_train_states = []
    all_train_gold_actions = []

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
            all_train_states.append(session_states)
            all_train_gold_actions.append(session_gold_actions)

    all_train_states = flatten(all_train_states)
    all_train_gold_actions = flatten(all_train_gold_actions)
    print('num state and actions: ', len(all_train_states))

    # dialogue act classifier 
    dialog_act_classifier = DialogueActClassifier()

    # start training
    total_cost = 0
    num_state_action_pairs = len(all_train_states)
    play_outcomes = []
    for i in tqdm(list(range(num_state_action_pairs))):
        cur_state = all_train_states[i]
        cur_gold_action = all_train_gold_actions[i] 
        play_result = play_state(
            cur_state, 
            cur_gold_action, 
            dialog_act_classifier = dialog_act_classifier,
            max_align_loop = args.max_align_loop,
            sender_lm_name = args.sender_lm_name,
            receiver_lm_name = args.receiver_lm_name
            )
        play_outcomes.append(play_result)
        total_cost += play_result['cost']
        print(f'Total Cost So Far: {total_cost}')

    # saving
    pd.DataFrame(play_outcomes).to_csv(args.output_csv_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training the Agent')
    parser.add_argument('--data_path', type=str, default=None,
                        help='csv to the training data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='csv to the order of dialog ids to be used for training data')
    parser.add_argument('--output_csv_file', type=str, default=None,
                        help='csv file to output the summarized rules')
    parser.add_argument('--sender_lm_name', type=str, default='gpt-3.5-turbo-1106',
                        help='openai name of the sender lm')
    parser.add_argument('--receiver_lm_name', type=str, default='gpt-3.5-turbo-1106',
                        help='openai name of the receiver lm')
    parser.add_argument('--max_align_loop', type=str, default=3,
                        help='maximum step of alignment loops to be used')
    parser.add_argument('--num_dialogues_to_use', type=int, default=5,
                        help='how many dialogue to use, if you have more than you want')
    parser.add_argument('--begin_dialogue_position', type=int, default=0,
                        help='where to begin, e.g. if you set this to 5, your dialogues use will be [5:5+num_dialogues_to_use]')
    args = parser.parse_args()
    print(args)
    main(args)
    
    
    