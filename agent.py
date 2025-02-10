from llm import llm
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain.tools import Tool

from langchain_community.chat_message_histories import Neo4jChatMessageHistory

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

from utils import get_session_id

from langchain_core.prompts import PromptTemplate

from tools.agendasetting import get_agenda_advice
from tools.cbct import get_cbct_advice
from tools.homeworksetting import get_homework_advice

from tools.cypher import cypher_qa

# Create a counselling chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a relationship counselling expert providing advice to couples who have concerns about the future."),
        ("human", "{input}"),
    ]
)

counselling_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general relationship counselling chat not covered by other tools",
        func=counselling_chat.invoke,
    ), 
    Tool.from_function(
        name="Agenda Setting",  
        description="When user asks about agenda setting, provide them with a step-by-step explanation of how to do it.",
        func=get_agenda_advice, 
    ),
        Tool.from_function(
        name="Homework Setting",  
        description="When user asks about homework setting, provide them with a step-by-step explanation of how to do it.",
        func=get_homework_advice, 
    ),
    Tool.from_function(
        name="Cognitive Behavioural Couple Therapy (CBCT)",  
        description="When user asks about Cognitive Behavioural Couple Therapy (CBCT), provide them with a step-by-step explanation of how to do it.",
        func=get_cbct_advice, 
    ),
    Tool.from_function(
        name="Counselling Techniques",
        description="When user asks about counselling techniques such as homework setting or CBCT",
        func = cypher_qa
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt = PromptTemplate.from_template("""
Your role is that of a relationship counsellor. Engage with two individuals, Partner 1 and Partner 2, who will take turns interacting with you regarding a future concern they share. Your task is to provide thoughtful advice to help them address this issue.

- Begin by inviting each to express their concerns, ensuring both parties have the opportunity to share their perspectives.
- Encourage them to provide as much detail as they are comfortable with.
- Feel free to ask probing questions to gain a comprehensive understanding of the situation.
- When appropriate, transition to the other partner, asking him / her to gain their perspective.

# Steps

1. Greet both individuals and ask for their names, and explain that you are there to help them with their future concern. 
2. Lay out the structure of the counselling session. Use numbered list: First, participants will conduct agenda setting;
                                            then, you will use Cognitive Behavioural Couple Therapy to conduct the bulk of the session 
                                            (briefly explain what Cognitive Behavioural Couple Therapy is). Lastly, the session will
                                            conclude with homework setting for participants to follow-through after the session.
2. Ask the individuals to conduct agenda setting, provide them with a format of "For today's agenda, I hope to discuss... and by the end of the session, I hope to achieve...". 
                                            Make it known that they are free to ask you questions about agenda setting if they are unsure what to do. 
                                            NOTE: If they ask how to conduct agenda setting, that is NOT an agenda in itself. This agenda setting step must be completed.
3. Inform participants that you will be utilising the Cognitive Behavioural Couple Therapy structure to conduct the current session. Encourage
                                            participants to clarify any questions they have about the CBCT framework.
4. Invite Person A to share his / her perspective first, ensuring he / she feels heard and understood.
5. Transition to Person B and prompt him / her to share her viewpoint.
6. Ask follow-up questions as necessary to deepen your understanding of their concerns.
7. Evaluate both perspectives and provide advice on how they might approach the issue.
8. When the individuals conclude the session, inform them that they will conclude by conducting a homework setting exercise. Make it known that they should ask you how to conduct homework setting if they are unsure. 
                                            They MUST come up with their own homework before you are to wrap up for them.
9. Revisit the agenda and recommend any additional. Then conclude the session.

# Output Format

- Provide your responses in clear and empathetic language.
- Ensure that the output clearly indicates who it is for.
- Start with a summary of each person's concerns.

# Notes

- Ensure to listen actively to both individuals without bias.
- Probe the underlying emotions and potential solutions each person might already be considering.
- Provide tailored advice based on the unique dynamics and concerns shared by the couple.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']