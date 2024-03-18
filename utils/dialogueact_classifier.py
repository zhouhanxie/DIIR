from .assets import annomi_classifier_path
import openai 
from .classification_wrapper import ClassificationWrapper
from .openai_dialogue import OpenaiSequencialDialogue
from nltk import sent_tokenize



class DialogueActClassifier:

    """
    Dialogue Act Classification, supericl style (small model + LLM)
    """

    def __init__(self):
        self.intent_classifier = ClassificationWrapper(
            model_name_or_path = annomi_classifier_path
        )
        
        
    
    def annotate_dialogue_turn(self, context, turn):
        """
        Annotate the whole dialogue.

        Context:
            the context.
        Turn:
            the turn
        Sentence:
            sentences in that turn, splitted by nltk.sent_tokenize
        """
        out = []
        sentences = []
        tot_cost = 0
        for sent in sent_tokenize(turn):
            sentences.append(sent)
            label, cost = self.classify_action(context, turn, sent)
            out.append(label)
            tot_cost += cost
        return sentences, out, tot_cost
    
            
    def get_sorted_labels(self, sentence):
        p = self.intent_classifier(sentence, return_type='class proba', disable_tqdm=True)
        sorted_labels = sorted(p['predicted'].keys(), key=lambda k : p['predicted'][k], reverse=True)
        return sorted_labels

    def classify_action(self, context, turn, sentence, verbose=False):
        """
        Context: the dialogue context
        Turn: the turn of interest in the dialogue
        Sentence: the sentence of interest in the turn
        """
        classifier_prompt = """Look at the following dialogue snippet, and tell me the intent of a therapist response. The potential labels and their explanations are:

    [Give Information]: Gives information, educates, provides feedback, or expresses a professional opinion without persuading, advising, or warning. Self-discose of objective information also goes here.
    [Question]: All questions from clinicians (open, closed, evocative,	fact-finding, etc.)	
    [Simple Reflection]: Reflect (repeat or reword) on what the client have said, without adding further meaning to it.
    [Complex Reflection]: Reflect (repeat or reword) on what the client have said, but adding further meaning (or make explicit some hidden impliciation) of it.
    [Affirm]: States something positive or complimentary about the client’s strengths, efforts, intentions, or worth.
    [Emphasize Autonomy]: Highlights a client’s sense of control, freedom of choice, personal autonomy, ability, and obligation about change.
    [Confront]: Directly and unambiguously disagreeing, arguing, correcting, shaming, blaming, criticizing, labeling, warning, moralizing, ridiculing, or questioning a client’s honesty.
    [Seek Collaboration]: Attempts to share power or acknowledge the expertise of a client.
    [Support]: These are generally sympathetic, compassionate, or understanding comments, with the quality of siding with the client.
    [Advise with Permission]: Attempts to change a client’s opinions, attitudes, or behaviors, but have obtained the client's permission to do so, or clearly indicates the decision is the clients'.
    [Advise]: Attempts to change a client’s opinions, attitudes, or behaviors using tools such as logic, compelling arguments, self-disclosure, facts, biased information, advice, suggestions, tips, opinions, or solutions to problems.
    [Other]: Filler words, such as 'mm-hmm', 'mm', 'yeah', 'okay', 'hmm', 'uh-huh', 'huh', 'right', 'yep', etc.


    The snippet:
    @snippet@

    What is the label of the sentence "@sentence@" in the last therapist response? Our classifier classified this as @classifier_decision@, now help use decide which best describes this sentence. You can override the classifier decision if you really need to, but 95% times the true answer is in the classifier's suggestion. 

    Your answer (answer with the label only, without any extra words):
        """
        classifier = OpenaiSequencialDialogue(model='gpt-3.5-turbo-1106')
        # small_model_decision = self.intent_classifier(sentence, disable_tqdm=True)['predicted']
    #     if 'advise' in small_model_decision.lower():
    #         small_model_decision = 'Advise (but unsure about whether with permission or not)'
    #     if 'reflection' in small_model_decision.lower():
    #         small_model_decision = 'Reflection (but unsure about whether it is simple or complex)'
        small_model_decision = self.get_sorted_labels(sentence)
        small_model_decision = ' likely one of '+str(small_model_decision[:5])+' '
        response = classifier.send_user_message(
            classifier_prompt.replace('@snippet@', context+'\n'+turn).replace('@sentence@', sentence).replace('@classifier_decision@', small_model_decision)
        )
        if verbose:
            print(classifier_prompt.replace('@snippet@', context+'\n'+turn).replace('@sentence@', sentence).replace('@classifier_decision@', small_model_decision))
            print(response)
        return response.replace('[', '').replace(']', ''), classifier.cost()['total']
    
if __name__ == '__main__':
    from AligningMotivationalInterviewing.utils.assets import OPENAI_API_KEY
    import openai
    openai.api_key = OPENAI_API_KEY
    dialogue_act_classifier = DialogueActClassifier()
    dialogue_context = """Topic: smoking cessation 
[client]: Yeah. Sure.
[therapist]: Okay, good. So we can go over your laboratory results at that time, as well as review your plan. Now, would you like some information on smoking cessation?
[client]: Yeah, I-I think that would be very helpful."""
    dialogue_turn = """[therapist]: glad you think that's helpful! I'm here to help you with these things. Remember, you have the power to resolve this!"""
    print(dialogue_context)
    print(dialogue_turn)
    print('sentences, labels, cost: ', dialogue_act_classifier.annotate_dialogue_turn(dialogue_context, dialogue_turn))