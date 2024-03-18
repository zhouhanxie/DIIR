def display_annomi_dialogue(annomi_dataset, dialogue_id, turn_sep='\n',max_num_turns=100):
    dialogue_turns = annomi_dataset[annomi_dataset['transcript_id'] == dialogue_id].drop_duplicates(subset='timestamp').to_dict('records')
    dialogue_turns = sorted(dialogue_turns, key=lambda x:x['timestamp'])
    topic = dialogue_turns[0]['topic']
    output = """"""
    num_turns_sofar = 0
    
            
    i = 0
    while i < len(dialogue_turns):
        
        output += '['+dialogue_turns[i]['interlocutor']+']: '
        
        
        while (i < len(dialogue_turns)) and dialogue_turns[i]['utterance_text'].endswith('-'):
            output += dialogue_turns[i]['utterance_text']
            i += 2
        
         
        if not (i < len(dialogue_turns)):
            break
        output += dialogue_turns[i]['utterance_text']+turn_sep
        i += 1
        num_turns_sofar += 1
        if num_turns_sofar >= max_num_turns:
            break
            
    topic = dialogue_turns[0]['topic']
    turns = []
    for t in output.split(turn_sep):
        turn_content = t.split(']: ')
        role = turn_content[0][1:]
        utt = ' '.join(turn_content[1:])
        turns.append({'role':role, 'utterance':utt})
    
            
    return {'topic':topic, 'turns':turns}