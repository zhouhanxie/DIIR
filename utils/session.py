

# def assemble_dialogue_turns(dialogue_turns, turn_sep='\n',max_num_turns=100):
#     output = """"""
    
#     for turn in dialogue_turns:
#         output += '['+turn['role']+']: '
#         output += turn['utterance']+turn_sep
        
#     return output

def is_simple_acknowledgement(turn):
    """
    rule based analysis to determine whether the therapist is simply acknowledging the client
    """
    simple_ack_units = ['mm-hmm', 'mm', 'yeah', 'okay', 'hmm', 'uh-huh', 'huh', 'right', 'yep']
    simple_ack_units = simple_ack_units + [u+'.' for u in simple_ack_units]
    simple_ack_units = simple_ack_units + [u+',' for u in simple_ack_units]
    simple_ack_units = set(simple_ack_units)
    turn_elements = turn.strip().lower().split()
    for element in turn_elements:
        if element not in simple_ack_units:
            return False
    
    return True

def assemble_dialogue_turns(dialogue_turns, turn_sep='\n',max_num_turns=100):
    output = """"""
    
    prev_role = None
    for turn in dialogue_turns:
        if turn['utterance'] == '':
            continue
        if turn['role'] == 'therapist' and (is_simple_acknowledgement(turn['utterance']) or len(turn['utterance'])<15):
            # in case you want to do something here
            utt = turn['utterance']
            # continue
        else:
            utt = turn['utterance']
        if turn['role'] != prev_role:
            if prev_role != None:
                output+=turn_sep
            output += '['+turn['role']+']: '
            output += utt
            prev_role = turn['role']
        else:
            output = output + ' '+utt
        
    return output

class Session:
    
    def __init__(self, dialogue):
        self.topic = dialogue['topic']
        self.turns = dialogue['turns']
        self.state_tracker = 0
        
    def state(self, max_prev_turns=4):
        return 'Topic: '+self.topic+'\n'+assemble_dialogue_turns(
            self.turns[max(0, self.state_tracker-max_prev_turns):self.state_tracker]
        )
    
        
    
    def gold_action(self):
        return assemble_dialogue_turns([self.turns[self.state_tracker]]).strip()
    
    def skip_to_next_utt_by(self, role):
        try:
            self.state_tracker += 1
            while self.turns[self.state_tracker]['role'] != role:
                self.state_tracker += 1
            return self.gold_action()
        except IndexError:
            raise IndexError(f'No more utterance by {role}')
            
    def has_next_utt_by(self, role):
        try:
            cur = self.state_tracker+1
            while self.turns[cur]['role'] != role:
                cur += 1
            return True
        except IndexError:
            return False
        
    def full_session(self):
        return 'Topic: '+self.topic+'\n'+assemble_dialogue_turns(
            self.turns
        )
    

def assemble_state_with_instructions(raw_state, guideline=''):
    output = raw_state
    output += '\n'+ """Predict what the therapist would say."""
    return output