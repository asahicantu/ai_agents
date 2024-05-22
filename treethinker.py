import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama
# os.environ["SERPER_API_KEY"] = "Your Key"  # serper.dev API key
# os.environ["OPENAI_API_KEY"] = "Your Key"

OPENAI_API_BASE = 'http://localhost:11434/v1'
OPENAI_MODEL_NAME = 'llama3'  # Adjust based on available model
OPENAI_API_KEY = ''
search_tool = SerperDevTool()

# llm = ChatOpenAI(model=OPENAI_MODEL_NAME, base_url=OPENAI_API_BASE)
llm = Ollama(model=OPENAI_MODEL_NAME)

agents = {
    "Marketer":
    Agent(
        role="Market Research Analyst",
        goal="Find out how big is the demand for my products and suggest how to reach the widest possible customer base",
        backstory="""You are an expert at understanding the market demand, target audience, and competition. This is crucial for validating
         whether and idea filfills a market need adn has the potential to attract a wide audience.
         You are good at coming up with ideas on how to appeal to widest possible audience.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    ),
    "Technologist":
    Agent(
        role="Technologist",
        goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
        backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your 
		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of 
		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances 
		operational efficiency but also provides a competitive edge in the market.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    ),
    "BusinessConsultant":
    Agent(
        role="BusinessConsultant",
        goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
        backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas 
            into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams. 
            Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
            about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )
}

tasks = [
    Task(
        description="""Analyze what the market demand for plugs for holes in crocs (shoes) so that this iconic footware looks less like swiss cheese. 
		Write a detailed report with description of what the ideal customer might look like, and how to reach the widest possible audience.
    """,
        expected_output="A concise report with at least 10 bullet points addressing the most important areas when it comes to marketing this type of business",
        agent=agents["Marketer"]
    ),
    Task(
        description="""Analyze how to produce plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese. 
    """,
        expected_output="A detailed report with description of how the product can be produced, what type of materials are needed, and what the production process would look like.",
        agent=agents["Technologist"]
    ),

    Task(
        description="""Analyze and summarize marketing and technological report and write a detailed business plan with 
		description of how to make a sustainable and profitable "plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese" business.
    """,
        expected_output="A concise business report with at least 10  bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.",
        agent=agents["BusinessConsultant"]
    )
]


crew = Crew(
    agents=agents.values(),
    tasks=tasks,
    verbose=2,
    # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    process=Process.sequential,
)

result = crew.kickoff()

print("######################")
print(result)
