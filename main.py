from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

from textwrap import dedent
from agents import NewsletterAgents
from tasks import NewsletterTasks
from tools.file_write import FileWriteTool
from tools.directory_write import DirWriteTool
from crewai_tools import SerperDevTool, FileReadTool
from tools.data_viz import DataVisualizationTool

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize tools
web_search_tool = SerperDevTool()
file_read_tool = FileReadTool()

file_write_tool = FileWriteTool.file_write_tool
dir_write_tool = DirWriteTool.dir_write_tool
data_viz_tool = DataVisualizationTool.data_viz_tool

# Define tool sets for each agent
journalist_tools = [file_read_tool, file_write_tool]
editor_tools = [file_read_tool, file_write_tool]
researcher_tools = [web_search_tool, file_read_tool, file_write_tool]
data_analyst_tools = [file_read_tool, data_viz_tool, file_write_tool]
critic_tools = [file_read_tool, web_search_tool, file_write_tool]
seo_tools = [web_search_tool, file_read_tool, file_write_tool]
sme_tools = [web_search_tool, file_read_tool, file_write_tool]

class NewsletterCrew:
    def __init__(self, topic):
        self.topic = topic

    def run(self):
        agents = NewsletterAgents()
        tasks = NewsletterTasks()

        # Initialize agents
        journalist_agent = agents.climate_journalist_agent(journalist_tools)
        editor_agent = agents.editor_agent(editor_tools)
        researcher_agent = agents.researcher_agent(researcher_tools)
        data_analyst_agent = agents.data_analyst_agent(data_analyst_tools)
        critic_agent = agents.critic_agent(critic_tools)
        seo_agent = agents.seo_specialist_agent(seo_tools)
        sme_agent = agents.subject_matter_expert_agent(sme_tools)

        # Define tasks
        research_task = tasks.research_task(researcher_agent, researcher_tools, self.topic)
        writing_task = tasks.writing_task(journalist_agent, journalist_tools, [research_task], self.topic)
        data_analysis_task = tasks.data_analysis_task(data_analyst_agent, data_analyst_tools, [research_task])
        editing_task = tasks.editing_task(editor_agent, editor_tools, [writing_task], self.topic)
        sme_review_task = tasks.expert_review_task(sme_agent, sme_tools, [editing_task])
        critique_task = tasks.critique_task(critic_agent, critic_tools, [sme_review_task], self.topic)
        seo_task = tasks.seo_optimization_task(seo_agent, seo_tools, [critique_task], self.topic)

        crew = Crew(
            agents=[journalist_agent, editor_agent, researcher_agent, data_analyst_agent, critic_agent, seo_agent, sme_agent],
            tasks=[research_task, writing_task, data_analysis_task, editing_task, sme_review_task, critique_task, seo_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result

# FastAPI app
app = FastAPI()

class NewsletterRequest(BaseModel):
    topic: str

@app.post("/generate-newsletter")
def generate_newsletter(request: NewsletterRequest):
    try:
        crew = NewsletterCrew(request.topic)
        result = crew.run()
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
