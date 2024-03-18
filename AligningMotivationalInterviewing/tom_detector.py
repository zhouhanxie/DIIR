from utils import OpenaiSequencialDialogue 

def determine_therapist_stage(dialog_snippet, llm_name='gpt-3.5-turbo-1106'):
    prompt = """You are a dialogue analyst and your job is to help us understanding motivational interviewing dialogues.  You will be given a dialogue context, and you will help us determine which of the 4 stages of motivational interviewing the therapist is at: engaging, focusing, evoking, or planning.

1. **Engaging**: This is the first process, where the goal is to establish a trusting and respectful relationship between the therapist and the client. Engaging involves active listening and expressing empathy to create a comfortable environment that encourages the client to open up and talk about their experiences and issues. Effective engagement sets the stage for the work to come.

2. **Focusing**: During the focusing process, the therapist helps the client to determine the direction of the conversation and identify what changes are important and possible to work on. It involves clarifying the agenda and honing in on specific areas that may benefit from change. Setting goals and priorities is a key part of the focusing stage.

3. **Evoking**: In this stage, the therapist's role is to elicit the client's own motivations for change. The client is encouraged to talk about their desires, abilities, reasons, and need for change (known as DARN). The therapist uses reflective listening and open-ended questions to draw out the client’s personal reasons for change, their understanding of the issue, and any ambivalence they may feel.

4. **Planning**: The final process involves developing a commitment to change and formulating a concrete plan of action. The therapist collaborates with the client to create strategies and steps to initiate and sustain change. This stage includes setting goals, considering options, discussing the pros and cons of different strategies, and planning for potential obstacles.

Look at the following dialogue snippet, which of the 4 stage is the dialogue in?

@snippet@

Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything."""
    model = OpenaiSequencialDialogue(model=llm_name)
    model_response = model.send_user_message(prompt.replace('@snippet@', dialog_snippet))
    try:
        therapist_tom = eval(model_response)['prediction']
    except:
        print(model_response)
        therapist_tom = 'na'
    cost = model.cost()['total']
    
    return therapist_tom.lower(), cost

def determine_toms(
    dialog_snippet, 
    therapist_response,
    llm_name='gpt-3.5-turbo-1106'
):
    prompt = """You are a dialogue analyst and your job is to help us understanding motivational interviewing dialogues.  You will be given a dialogue context, and you will help us determine which of the 5 stages of change the client is at: Precontemplation, Contemplation, Preparation, Action, or Maintenance.

1. Precontemplation: At this stage, the individual is not yet considering making a change and may be unaware of the need for change.

2. Contemplation: In this stage, the individual is aware of the need for change and is considering the possibility of making a change in the near future.

3. Preparation: During this stage, the individual is actively preparing to make a change and may be taking small steps toward behavior change.

4. Action: At this stage, the individual has made a specific, observable change in their behavior and is actively working to maintain this change.

5. Maintenance: In the maintenance stage, the individual has successfully made the desired change and is working to prevent relapse and sustain the new behavior over time.

Look at the following dialogue snippet, which of the 5 stage is the client in?

@snippet@

Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything."""
    model = OpenaiSequencialDialogue(model=llm_name)
    model_response = model.send_user_message(prompt.replace('@snippet@', dialog_snippet))
    try:
        client_tom = eval(model_response)['prediction']
    except:
        print(model_response)
        client_tom = 'na'
        
    client_last_utt = dialog_snippet.split('\n')[-1].replace('[client]', '')
    fine_grained_tom_prompt = f"""Now, give a fine-grained description of the client's mental state when the client says """+client_last_utt+""" (it doesn't have to be within to the stages of change -  just your understanding of how ready they are to change/what they are thinking with respect to making a positive change - are they resistant? eager to change? hesitant?). Be general, make sure your rule is generalizable across topics. For example, simple use 'bad habit' instead of 'drug abuse/alcohol issue/smoking'. Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything."""
    model_response = model.send_user_message(fine_grained_tom_prompt)
    
    try:
        client_fine_grained_tom = eval(model_response)['prediction']
    except:
        print('[Warning]: TOM detector: parse response failed on: ', model_response)
        client_fine_grained_tom = model_response
    
    
    therapist_behavior_prompt = """Here is the therapist's response to the client:
@therapist_response@

Now, give a fine-grained description of the therapist's response. What is the therapist doing here? And what effect does the therapist want to have in the client using the response? Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything.
""".replace('@therapist_response@', therapist_response)
    model_response = model.send_user_message(therapist_behavior_prompt)
    try:
        therapist_fine_grained_tom = eval(model_response)['prediction']
    except:
        print(model_response)
        therapist_fine_grained_tom = 'na'
        
    contextual_instruction_prompt = """Now, based on your analysis of the client's mental state and the therapist's behavior, combine these two information into a rule in the format of "when the client ..., the therapist can… in order to…"""
    contextual_instruction = model.send_user_message(contextual_instruction_prompt)
    
    
    return client_tom.lower(), client_fine_grained_tom.lower(), therapist_fine_grained_tom.lower(), contextual_instruction, model.cost()['total']


def determine_toms_inference_mode(
    dialog_snippet, 
    llm_name='gpt-3.5-turbo-1106'
):
    prompt = """You are a dialogue analyst and your job is to help us understanding motivational interviewing dialogues.  You will be given a dialogue context, and you will help us determine which of the 5 stages of change the client is at: Precontemplation, Contemplation, Preparation, Action, or Maintenance.

1. Precontemplation: At this stage, the individual is not yet considering making a change and may be unaware of the need for change.

2. Contemplation: In this stage, the individual is aware of the need for change and is considering the possibility of making a change in the near future.

3. Preparation: During this stage, the individual is actively preparing to make a change and may be taking small steps toward behavior change.

4. Action: At this stage, the individual has made a specific, observable change in their behavior and is actively working to maintain this change.

5. Maintenance: In the maintenance stage, the individual has successfully made the desired change and is working to prevent relapse and sustain the new behavior over time.

Look at the following dialogue snippet, which of the 5 stage is the client in?

@snippet@

Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything."""
    model = OpenaiSequencialDialogue(model=llm_name)
    model_response = model.send_user_message(prompt.replace('@snippet@', dialog_snippet))
    try:
        client_tom = eval(model_response)['prediction']
    except:
        print(model_response)
        client_tom = 'na'
        
    client_last_utt = dialog_snippet.split('\n')[-1].replace('[client]', '')
    fine_grained_tom_prompt = f"""Now, give a fine-grained description of the client's mental state when the client says """+client_last_utt+""" (it doesn't have to be within to the stages of change -  just your understanding of how ready they are to change/what they are thinking with respect to making a positive change - are they resistant? eager to change? hesitant?). Be general, make sure your rule is generalizable across topics. For example, simple use 'bad habit' instead of 'drug abuse/alcohol issue/smoking'. Format you answer in this format: {'prediction': "your answer"}, you do not have to explain anything."""
    model_response = model.send_user_message(fine_grained_tom_prompt)
    
    try:
        client_fine_grained_tom = eval(model_response)['prediction']
    except:
        print('[Warning]: TOM detector: parse response failed on: ', model_response)
        client_fine_grained_tom = model_response
    
    
    return client_tom.lower(), client_fine_grained_tom.lower(), model.cost()['total']


if __name__ == '__main__':
    from utils import OPENAI_API_KEY
    import openai
    openai.api_key = OPENAI_API_KEY
    
    dialogue_context = """Topic: smoking cessation 
[client]: Yeah. Sure.
[therapist]: Okay, good. So we can go over your laboratory results at that time, as well as review your plan. Now, would you like some information on smoking cessation?
[client]: Yeah, I-I think that would be very helpful."""
    dialogue_turn = """[therapist]: glad you think that's helpful! I'm here to help you with these things. Remember, you have the power to resolve this!"""
    print(dialogue_context)
    print(dialogue_turn)
    print('client stage, client tom, therapist tom, contextual instruction, cost', determine_toms(dialogue_context, dialogue_turn))