## ReAct Agent ##################################
#      A ReAct (Reasoning + Acting) agent is an AI system that 
#    combines step-by-step chain-of-thought reasoning with 
#    external tool execution to solve complex problems. Instead 
#    of just generating answers, it follows a loop of thinking 
#    about a task, taking an action (e.g., searching the web, 
#    calculating), observing the results, and refining its approach.

# The ReAct agent operates in an iterative, three-step loop designed 
#   to mimic human problem-solving: 
#   - Thought: The LLM reasons about the current situation, analyzing what information is missing.
#   - Action: The agent selects and uses a tool (e.g., a search API, Python interpreter) to fetch data.
#   - Observation: The agent analyzes the feedback from the tool to decide if it has enough information or needs to repeat the loop.
#################################################
from nodes.image_or_joke
from nodes.text_to_image_gen import text_to_image
from nodes.joke_gen import image_or_joke
from nodes.nsfw_filter import nsfw_filter
from nodes.text_to_speech import text_to_speech as tts
from nodes.speechToText import transcribe_audio

from typing import Annotated, Literal, Union
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import MessagesState # comes prebuilt with a list of AnyMessage objects and add_messages reducer

class InputState(MessagesState):
    user_input: str
    system_prompt: str

class OutputState(MessagesState):
    agent_output: str

class OverallState(MessagesState):
    system_prompt: str
    user_input: str
    agent_output: str

class InputInterractionState(MessagesState):
    bar: str

def speech_to_text(state: InputState) -> InputInterractionState:
    text = transcribe_audio(prompt)
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def text_to_speech(state: InputState):
    speech = tts(input_prompt)
    return {"foo": state["user_input"] + " name"}

def nsfw_input_filter(state: InputInterractionState) -> InputInterractionState:
    filter_ = nsfw_filter()
    if os.path.isfile(input_prompt):
        result = filter_.filter_image(input_prompt, to_file=True)
    else:
        result = filter_.filter_text(input_prompt)
    # Read from OverallState, write to InputInterractionState
    return {"bar": state["foo"] + " is"}

def image_or_joke_filter(state: InputInterractionState) -> InputInterractionState:
    image_joke_filter = image_or_joke(input_prompt)
    # Read from InputInterractionState, write to InputInterractionState
    return {"agent_output": state["bar"] + " Lance"}

def llm_orchestrator(
        input_state: InputInterractionState,
        output_state: OutputInterractionState,
    ) -> Union[InputInterractionState, OutputInterractionState]:
    # Read from InputInterractionState, write to InputInterractionState
    return {"agent_output": state["bar"] + " Lance"}

def inappropriate_input_handler(state: InputInterractionState) -> InputInterractionState:
    ''' Input has been deemded innapropriate. If this is text from the user, 
        tell the user that their input is not appropriate and to try again with
        an appropriate joke or image request. If this is text from a model, tell 
        the model the same thing and run it back through. If this is an image from
        a model, apologize to the user and tell them that you couldn't generate 
        an image for the requested prompt.'''
        prompt_for_text = "Yout input is not appropriate. Please try again with \
            an appropriate joke or image request"
        prompt_for_image = "I apologize. I cannot generate \
            an image for your request."
        prompt_to_text_model = "Your respnse was innapropriate, try again with \
            something different."
        tts(prompt)
    # Read from InputInterractionState, write to InputInterractionState
    return {"agent_output": state["bar"] + " Lance"}

def invalid_input_handler(state: InputInterractionState) -> InputInterractionState:
    ''' Input prompt is not a joke or image request. Tell the user that
        It wasn't clear whether they requested a joke or an image and try again.'''
    prompt = "That didn't quite sound like a joke or picture request. Try again."
    tts(prompt)
    # Read from InputInterractionState, write to InputInterractionState
    return {"agent_output": state["bar"] + " Lance"}

def text_to_image_generator(state: InputInterractionState) -> OutputInterractionState:
    tti = text_to_image()
    result = tti.filter_text(input_prompt)
    # Read from InputInterractionState, write to OutputState
    return {"agent_output": state["bar"] + " Lance"}

def funny_joke_generator(state: InputInterractionState) -> OutputInterractionState:
    joke = image_or_joke(input_prompt)
    # Read from InputInterractionState, write to OutputState
    return {"agent_output": state["bar"] + " Lance"}

def nsfw_output_filter(state: OutputInterractionState) -> OutputInterractionState:
    filter_ = nsfw_filter()
    if os.path.isfile(input_prompt):
        result = filter_.filter_image(input_prompt, to_file=True)
    else:
        result = filter_.filter_text(input_prompt)
    # Read from InputInterractionState, write to OutputState
    return {"agent_output": state["bar"] + " Lance"}

def inbound_route_decision(
        input_state: InputInterractionState
    ) -> Literal[
        "text_to_image_generator", 
        "funny_joke_generator",
        "inappropriate_input_handler",
        "invalid_input_handler",
        ]:
    """Route based on whether input requests an image or a joke."""
    if input_state["joke_request"]:
        return Command(
            # state update
            update={"foo": "bar"},
            # control flow
            goto="funny_joke_generator"
        )
    elif input_state["image_request"]:
        return Command(
            # state update
            update={"foo": "bar"},
            # control flow
            goto="text_to_image_generator"
        )
    elif input_state["nsfw_input"]: # inappropriate input detected
        return "inappropriate_input_handler"
    else input_state["invalid_input"]: # not joke or image request
        return "invalid_input_handler"

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node(speech_to_text)
builder.add_node(nsfw_input_filter)
builder.add_node(image_or_joke_filter)
builder.add_node(llm_orchestrator)
builder.add_node(inappropriate_input_handler)
builder.add_node(invalid_input_handler)
builder.add_node(text_to_image_generator)
builder.add_node(funny_joke_generator)
builder.add_node(nsfw_output_filter)

# Inbound edges
builder.add_edge(START, "speech_to_text")
builder.add_edge("speech_to_text", "nsfw_input_filter")
builder.add_edge("nsfw_input_filter", "image_or_joke_filter")
builder.add_edge("image_or_joke_filter", "llm_orchestrator")
builder.add_conditional_edges("llm_orchestrator", route_decision)

# Outbound edges
builder.add_edge("text_to_image_generator", "nsfw_output_filter")
builder.add_edge("funny_joke_generator", "nsfw_output_filter")
builder.add_edge("nsfw_output_filter", "llm_orchestrator")
builder.add_edge("nsfw_output_filter", "llm_orchestrator")
builder.add_edge("llm_orchestrator","text_to_speech")
builder.add_edge("text_to_speech", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})