import logging
import sys
from typing import List, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from datetime import datetime
import streamlit as st
import wikiquote

# configs for script

USE_STREAMLIT = True
# Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = "Your API key"

# Initialize ChatOpenAI model
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-3.1-8b-instruct:free", temperature=0.75)

story_structure_chooser = """
use the handbook for choosing story structure to determine which story structure is proper for a story like this:
{story_prompt}


explain your reasoning througly and in depth, reference similar popular works and how they aided your decision to use the structure you choose.
here is the handbook:
Introduction
Selecting the right story structure is crucial for shaping a compelling narrative. Each structure offers unique benefits, guiding how your story unfolds and ensuring that it resonates with your audience. This handbook will help you understand various story structures and choose the one that best suits your narrative needs.



Classic Story Structure Overview: The classic story structure is a foundational model used in storytelling. It consists of five key stages that guide the narrative from beginning to end.


Sections:


Exposition: Introduces the protagonist, their world, and their desires. Ends with the inciting incident.
Rising Action: The protagonist faces challenges while pursuing their goal.
Climax: The protagonist confronts the main conflict head-on.
Falling Action: The consequences of the climax unfold.
Resolution: The story concludes, resolving character arcs and conflicts.
When to Use:
Ideal for traditional narratives, particularly in genres like romance, drama, or adventure.



Freytag’s Pyramid Overview: Named after 19th-century playwright Gustav Freytag, this structure emphasizes a tragic arc, often used in classical literature.


Sections:


Introduction: Establishes the protagonist's status quo and the inciting incident.
Rising Action: The protagonist pursues their goal, with stakes heightening.
Climax: A point of no return, leading to the protagonist's downfall.
Falling Action: The protagonist faces the aftermath of the climax.
Catastrophe: The protagonist reaches their lowest point, often ending in tragedy.
When to Use:
Best for tragic tales or stories with a somber tone, where the protagonist faces inevitable downfall.



The Hero’s Journey Overview: Popularized by Joseph Campbell and adapted by Christopher Vogler, this structure follows a hero's transformative journey.


Sections:


The Ordinary World: Introduces the hero's normal life.
Call to Adventure: The hero is presented with a challenge.
Refusal of the Call: The hero hesitates to take on the challenge.
Meeting the Mentor: A guide prepares the hero for their journey.
Crossing the Threshold: The hero enters a new, unfamiliar world.
Tests, Allies, Enemies: The hero faces challenges and makes allies.
Approach to the Inmost Cave: The hero nears their goal.
The Ordeal: The hero faces their greatest challenge.
Reward: The hero achieves their goal or gains something significant.
The Road Back: The hero returns home, often facing new challenges.
Resurrection: The hero undergoes a final transformation.
Return with the Elixir: The hero returns to their ordinary world, transformed.
When to Use:
Suitable for epic tales, fantasy, adventure, and stories where a protagonist undergoes significant transformation.



Three Act Structure Overview: A widely used framework that divides the narrative into three distinct parts: beginning, middle, and end.


Sections:


Act 1: Setup
Exposition: Establishes the protagonist's world.
Inciting Incident: Sets the story in motion.
Plot Point 1: The protagonist commits to the challenge.
Act 2: Confrontation
Rising Action: The protagonist encounters obstacles.
Midpoint: A twist that raises the stakes.
Plot Point 2: The protagonist faces a critical test.
Act 3: Resolution
Pre-Climax: The protagonist prepares for the final confrontation.
Climax: The protagonist faces their greatest challenge.
Denouement: The story concludes, revealing the outcome.
When to Use:
Ideal for stories with clear conflict and resolution, such as dramas, comedies, and action films.



Dan Harmon's Story Circle Overview: A simplified version of the Hero’s Journey, focusing on character development within a cyclical narrative.


Sections:


Comfort Zone: The protagonist's normal life.
Need/Want: The protagonist desires something.
Unfamiliar Situation: The protagonist steps out of their comfort zone.
Adaptation: The protagonist adjusts to new challenges.
Obtaining the Goal: The protagonist achieves what they wanted.
Paying the Price: The protagonist realizes the cost of their goal.
Returning to Familiar: The protagonist returns to their old world.
Change: The protagonist is changed by their journey.
When to Use:
Best for character-driven stories, especially in TV shows or episodic content where characters undergo gradual change.



Fichtean Curve Overview: A tension-filled structure that skips initial exposition, starting with rising action and building through a series of crises.


Sections:


Inciting Incident: The story begins with a significant event.
Rising Action/Multiple Crises: The protagonist faces a series of escalating challenges.
Climax: The protagonist confronts the story’s central conflict.
Falling Action: The aftermath of the climax.
Resolution: The story concludes with a new normal.
When to Use:
Perfect for stories with intense drama and suspense, such as thrillers or novels heavy in flashbacks.



Save the Cat Beat Sheet Overview: Developed by Blake Snyder, this structure is prescriptive, with specific beats that occur at precise points in the narrative.


Sections:


Opening Image: Sets the tone of the story.
Setup: Establishes the protagonist's world and desires.
Theme Stated: Hints at the story's deeper meaning.
Catalyst: The inciting incident.
Debate: The protagonist hesitates before taking action.
Break into Two: The protagonist begins their journey.
B Story: A subplot that supports the main theme.
Fun and Games: The story delivers on its premise.
Midpoint: A twist that raises the stakes.
Bad Guys Close In: The protagonist faces increasing challenges.
All Is Lost: The protagonist hits rock bottom.
Dark Night of the Soul: The protagonist reflects and regroups.
Break into Three: The protagonist makes a final attempt to succeed.
Finale: The protagonist confronts the main conflict.
Final Image: Reflects the protagonist’s transformation.
When to Use:
Excellent for structured narratives like screenplays or novels that require tight pacing and clear turning points.



Seven-Point Story Structure Overview: Focuses on the highs and lows of a narrative, encouraging writers to start with the ending and work backward.


Sections:


The Hook: Establishes the protagonist’s initial state.
Plot Point 1: The inciting incident that sets the story in motion.
Pinch Point 1: A setback that increases tension.
Midpoint: The protagonist takes control of their fate.
Pinch Point 2: Another major setback.
Plot Point 2: The protagonist discovers a solution.
Resolution: The story’s main conflict is resolved.
When to Use:
Ideal for stories focused on dramatic transformations, especially in genres like fantasy, sci-fi, or adventure.

Choosing the right story structure is about aligning your narrative's needs with the strengths of each framework. Use this handbook to guide your decision, ensuring that your story is engaging, well-paced, and emotionally resonant.

"""

summarize_story_structure = """
summarize structure analysis and only describe what structure should be used and how. do not tell why
here is the structure analysis
{story_structure}
"""

blueprint_prompt = """
You are an advanced story creation assistant designed to help writers craft compelling narratives. Your task is to generate a long and detailed story outline, including settings, characters, world-building, and timeline. Afterward, you will provide a fitting title for the story and then list out multiple chapters from start to finish. Each chapter should include a title, a long and detailed synopsis of what happens, and a note on how it connects to the next chapter. The chapter outlines should describe the actual events, key developments, and turning points with careful attention to story structure. Finally, you will offer writing advice on dialogues, story writing techniques, and adherence to story structure. This will all be achieved in response to the user's prompt about the story's theme or subject.


Synopsis:


Story Synopsis:
Begin by crafting an intriguing synopsis that summarizes the story in a way that hooks the reader. This should resemble the synopsis found on the back of a book, giving an overview of the central plot, main conflicts, and the stakes involved. The synopsis should be descriptive, engaging, and leave the reader wanting to know more.
Detailed Story Outline:


Story Theme and Core Concept:


Describe the central theme of the story. What is the core concept around which the narrative revolves?
Identify the key messages or moral lessons intended for the reader.
Setting:


Provide a detailed description of the story’s setting(s). Include information about the geographical locations, time periods, and significant environmental or cultural elements.
If applicable, outline the political, social, and economic conditions that define this world.
World-Building:


If the story takes place in a fictional or fantastical world, describe its history, key events, myths, and lore.
Describe the rules of magic, technology, or any other unique systems within the world.
Include details on different species, races, or civilizations that inhabit the world.
Character Profiles:


Generate detailed profiles for the main and significant secondary characters.
Include their background, motivations, key relationships, and character arcs.
Mention how each character’s development will influence the story.
Timeline:


Provide a chronological timeline of major events leading up to the story’s beginning.
Outline key events within the story that drive the plot forward.
Story Title:


Based on the details provided above, suggest a creative and fitting title for the story.
Chapter Breakdown:


Chapter Titles and Synopses:


List the titles of each chapter in the story.
For each chapter, write a long and detailed synopsis of what happens. The chapter outlines should describe the actual events, key developments, and turning points in the story with careful attention to story structure. Ensure each chapter’s synopsis includes character actions, emotional beats, conflicts, and resolutions, providing a clear sense of progression.
Indicate how each chapter logically connects to the next, ensuring smooth story progression and maintaining coherence in the narrative structure.
Chapter Structure and Pacing:


Offer advice on maintaining the pacing and structure within each chapter.
Highlight the importance of balancing action, dialogue, and exposition to keep the story engaging.
Writing Advice:


Dialogue Writing:


Provide tips on how to write realistic and impactful dialogues.
Explain how dialogues can be used to reveal character traits, advance the plot, and build tension.
Story Structure:


Offer guidance on maintaining a coherent and engaging story structure.
Discuss common narrative structures (e.g., three-act structure, hero’s journey) and how they can be applied to the story.
Character Development:


Explain the importance of character arcs and how they should evolve throughout the story.
Offer techniques for making characters feel multi-dimensional and relatable.
Maintaining Reader Engagement:


Provide strategies for keeping the reader engaged throughout the story.
Discuss the use of cliffhangers, foreshadowing, and plot twists.


here is the suggested story structure:
{story_structure}


here is the prompt for the story:
{story_prompt}


using advices above now write out the blueprint for the story
"""

chapter_json_prompt = """
You are an advanced story analysis assistant. Your task is to take a detailed story blueprint, which includes a list of chapters with their corresponding descriptions, and convert it into a JSON object. Each chapter title should correspond to its chapter description in the JSON format.

Instructions:

Extract Chapter Information:

Review the provided story blueprint and identify the chapters and their corresponding descriptions.
For each chapter, ensure that the title and description are accurately paired.
Convert the extracted chapter information into a JSON object. Each chapter title should be the key, and the corresponding description should be the value.
JSON Format Example:
{{
  "chapters": 
    {{
      "The Awakening": "In this chapter, the protagonist wakes up in a strange world with no memory of how they arrived. They encounter a mysterious guide who hints at a grand quest that lies ahead.",
      "Journey Begins": "The protagonist sets out on their journey, facing their first set of challenges. Along the way, they meet allies who will join them in their quest.",
      "The Hidden Threat": "Unbeknownst to the protagonist, a dark force is tracking their every move. This chapter introduces the antagonist and the looming danger that will test the hero's resolve.",
      "Clash at the Crossroads": "The protagonist and their allies face a critical battle at a crossroads. This battle tests their strength and unity, setting the stage for the larger conflict to come.",
      "Revelation": "In this pivotal chapter, a shocking truth about the protagonist’s past is revealed, altering the course of their journey and forcing them to reconsider their mission.",
      "The Final Stand": "The story reaches its climax as the protagonist faces the antagonist in a final showdown. The stakes are higher than ever, and the outcome will determine the fate of the world.",
      "New Dawn": "The story concludes with the aftermath of the final battle. The protagonist reflects on their journey, and the world begins to heal. A sense of hope for the future is established."
  }}
}}

Output Requirements:
Ensure the JSON object is properly formatted, with each chapter title as a unique key and the chapter description as the corresponding value.
Verify that all chapters from the blueprint are included in the JSON output.
when writing the json make sure that the chapter titles doesn't include the chapter name or number. just the title provided for chapter. for example "chapter 1: the awakening" or "1 - awakening" is incorrect and only "awakening: is correct

here is the story blueprint
{story_blueprint}


now generate the chapters json, only the json with no pretext or post text
"""

act_generator_prompt = """
You are an advanced story structuring assistant. Your task is to take a given chapter and its blueprint, and break it down into a detailed three-act structure. Each act should be described in two full paragraphs, elaborating on the key events, character actions, and story progression. This breakdown serves as both a narrative guide and a writing roadmap. Each act's structure and focus should naturally emerge from the chapter's needs. While the primary focus is on crafting detailed descriptions, include concise writing advice on dialogue, pacing, and character interactions where appropriate.
here is the general story blueprint:
{story_blue_print}
and we are writing the acts of the chapter {chapter_number} with title {chapter_title}. it is about:
{chapter_desc}

Instructions:

Act Descriptions:
For each act, provide a detailed, two-paragraph description that thoroughly explains what happens, focusing on key events, character actions, and how the story progresses. Allow the narrative needs of the chapter to dictate the structure and focus of each act.
Although writing advice should be brief, ensure it addresses crucial elements such as dialogue, pacing, and character dynamics, guiding the writer on how to effectively convey the scene.
Output Example:

Here’s how the LLM should format the response based on a sample chapter:

Chapter Title: The Awakening

Chapter Blueprint:
In this chapter, the protagonist wakes up in a strange world with no memory of how they arrived. They encounter a mysterious guide who hints at a grand quest that lies ahead.

Three-Act Structure:

Act 1
Alex awakens in a dense, fog-covered forest, disoriented and unable to recall how they arrived. The scene is one of eerie quiet, with only the distant murmur of water breaking the silence. As Alex stumbles through the mist, they come across a flowing stream, which seems to be the only sign of life in this otherwise desolate place. The tension builds as Alex explores further, driven by a mix of fear and curiosity. It is at the stream’s edge that they encounter Seraphine, a figure draped in a dark, hooded cloak. Seraphine’s presence is as unsettling as it is intriguing; she speaks in riddles, offering hints about a grand quest but leaving Alex with more questions than answers. Their conversation is brief, but laden with subtext—Seraphine seems to know more about Alex than she reveals, and her cryptic words suggest that the path ahead will be fraught with challenges.

The interaction with Seraphine serves as a catalyst for the chapter, pushing Alex towards a journey that they do not fully understand. The act concludes with a palpable sense of unease, as Alex realizes they are far from home and bound to a destiny they cannot yet comprehend. The mist, symbolic of Alex’s confusion, only thickens as Seraphine disappears into the shadows, leaving Alex to ponder the weight of her words. This opening act sets the tone for the chapter, establishing the mysterious world and the beginning of Alex’s internal and external journey.

Writing Advice:
Keep dialogue minimal but impactful, allowing subtext to convey the underlying tension. Focus on descriptive language to build atmosphere and immerse the reader in Alex’s disoriented state.

Act 2
The narrative tension escalates as Alex ventures deeper into the forest, driven by an uneasy mix of dread and determination. The fog thickens around them, distorting the landscape and making every step feel like a leap into the unknown. Suddenly, the tranquility is shattered by the rustling of leaves and the emergence of shadowy creatures from the underbrush. These creatures, formless and dark, represent the first real threat in this new world, and their attack is swift and disorienting. Seraphine’s sudden reappearance is equally jarring; she instructs Alex to defend themselves, her tone urgent and unyielding. Alex’s initial attempts at self-defense are clumsy and ineffective, revealing their lack of experience and deepening their sense of vulnerability.

As the creatures close in, Seraphine urges Alex to focus, pushing them to tap into a power they didn’t know they possessed. Desperation fuels Alex’s actions, and in a moment of intense concentration, they manage to summon water from the stream, creating a barrier that momentarily holds the creatures at bay. This newfound ability surprises both Alex and Seraphine, though the latter’s reaction is one of careful observation rather than shock. The act ends with the creatures retreating into the shadows, leaving Alex exhausted but alive. This confrontation not only tests Alex’s physical limits but also hints at the latent powers that will become crucial in the chapters to come.

Writing Advice:
Use action sequences to reveal character strengths and weaknesses. Dialogue during combat should be terse and functional, focusing on the urgency of the situation. Balance descriptive action with moments of introspection to maintain narrative depth.

Act 3
With the immediate threat neutralized, the forest begins to transform—fog lifts slightly, revealing a narrow path that winds deeper into the unknown. The air is still tense, but there is a moment of calm as Alex catches their breath and Seraphine offers a few final words of guidance. Seraphine’s demeanor remains enigmatic; she speaks of the trials ahead and the importance of trusting one’s instincts over fragmented memories. Her words are less about comfort and more about preparation, underscoring the gravity of the journey Alex is about to undertake. There is a sense of a looming challenge, a foreshadowing of the dangers that lie ahead, and the choices Alex will have to make.

As Seraphine vanishes once more into the forest, Alex is left alone, standing at the edge of the path that has suddenly appeared. This act focuses on reflection and resolution, allowing Alex to process the events that have just unfolded and what they might signify. The chapter closes with Alex stepping onto the path, a symbolic act that marks the beginning of their quest. The resolution is quiet but charged with anticipation, setting the stage for the challenges that await in the next chapter.

Writing Advice:
End the chapter with a reflective tone, using dialogue to foreshadow future events without giving too much away. Create a smooth transition from action to introspection, maintaining the reader’s engagement through character-driven narrative.

Output Requirements:

Ensure each act is described in two full paragraphs, elaborating on the events, character actions, and story progression.
Provide brief but insightful writing advice on key elements such as dialogue, pacing, and character dynamics.

using above structure now generate three acts for chapter {chapter_number}: {chapter_title}
"""

acts_json = """
You are an advanced language model tasked with converting detailed written acts into a structured JSON format. Each act of the chapter should be represented as a key-value pair in the JSON object. The keys should be labeled as "act-1", "act-2", and "act-3", corresponding to the description of each act. Your goal is to take the provided descriptions and accurately format them into JSON.

Instructions:

Input
You will be provided with detailed descriptions of three acts from a chapter.

Output
the acts include description and writing advice for each act, write them down in json format

here are the acts:
{acts}

example output:
{{
    "act-1": {{ "description": "act-1 description", "writingAdvice": "act-1 writing advice"}},
    "act-2": {{ "description": "act-2 description", "writingAdvice": "act-2 writing advice"}},
    "act-3": {{ "description": "act-3 description", "writingAdvice": "act-3 writing advice"}},
}}
"""

write_act_prompt = """
You are an advanced story-writing assistant. Your task is to take a detailed act description and craft a compelling manuscript for that act. You will write the narrative, dialogue, and action sequences, ensuring the story is engaging, immersive, and believable. The manuscript should feel natural, with characters speaking and acting in ways that are consistent with their personalities and the story’s tone.

for your reference i provide some text from famous works similar to this story so you can take inspiration from:
{famous_work_reference}

here is what happened in previous chapters:
{chapters_summery}

Instructions:

Input
You will be provided with:
the prompt of the story
A story blueprint that outlines the overall plot, setting, and character motivations.
A detailed act description that outlines key events and interactions within the act.
Output
Using the information provided, write the actual manuscript (which is very long, at least 3 pages) for the act, including:

Engaging narration that vividly describes the setting, actions, and emotions.
Natural dialogue that reflects the characters’ personalities, relationships, and the situation they are in.
Believable pacing that maintains the reader’s interest and drives the story forward.
Writing Tips:

Show, Don’t Tell: Use descriptive language to paint vivid scenes and convey emotions. Instead of stating how a character feels, show it through their actions, expressions, and dialogue.
Dynamic Dialogue: Ensure that conversations between characters feel real and contribute to character development or plot progression. Avoid overly formal or stilted language unless it fits a character’s personality.
Pacing: Balance action, dialogue, and description to keep the story engaging. Vary sentence lengths and structures to create rhythm and momentum.
Character Consistency: Keep characters’ voices, behaviors, and decisions consistent with their established traits and motivations. Refer back to the story blueprint as needed.
Emotion and Tension: Infuse scenes with appropriate emotions and tension, using inner monologue and subtle details to deepen the reader’s connection to the characters.

now here is the story blueprint:
{story_blueprint}

here is the original prompt for this story:
{original_prompt}

here is the what chapter is about:
{chapter_desc}

here is the description of this act:
{act_description}

here are some advice for writing this act:
{act_writing_advice}

we are wring the act {act_number} of chapter {chapter_number}, try writing a great story for this act (should be long at least 3 pages). nowhere in the text mention that we are on act {act_number} just output the story text
only give the act text with NO pre text or post text such as "here is the text for act ..." or "anything else i can help with?", this allows the acts to be stitched together in future
"""


write_act_extra = """
here is what happened previously in this chapter:
{previous_acts}
we are wring the act {act_number} of chapter {chapter_number}, try writing a great story for this act (should be long at least 3 pages). nowhere in the text mention that we are on act {act_number} just output the story text
only give the act text with NO pre text or post text such as "here is the text for act ..." or "anything else i can help with?", this allows the acts to be stitched together in future

here is the ending lines of what happened previously. pick up where it is left up with no pre text:
{lasting_lines}... (now continue from here)
"""

popular_works_prompt = """
Extract the names of popular works mentioned in the given structure. Only include the title of the work, removing any additional descriptions or explanations.

For example, if the structure contains:
* Cinderella (Disney's classic fairy tale)
* Beauty and the Beast (Disney's classic fairy tale)
* The Princess Bride (a fantasy romance novel and film)
* The Hunger Games (a dystopian romance series)

You should extract:
["Cinderella", "Beauty and the Beast", "The Princess Bride", "The Hunger Games"]

Structure:
{structure}

Extract the popular works and format them as a JSON object with a single property "works" containing a list of strings.
"""

summarize_story_part = """
summarize this part of the story. while summarizing make sure to mention the all the character names and their associations. including the new characters introduced.

{story_part}
"""



# Define data structures for parsing
class Chapter(BaseModel):
    title: str = Field(description="Title of the chapter")
    description: str = Field(description="Description of the chapter")

class Chapters(BaseModel):
    chapters: dict[str, str] = Field(description="Dictionary of chapter titles and descriptions")

class Act(BaseModel):
    description: str = Field(description="Description of the act")
    writing_advice: str = Field(description="Writing advice for the act")

class Acts(BaseModel):
    act_1: Act = Field(description="First act of the chapter")
    act_2: Act = Field(description="Second act of the chapter")
    act_3: Act = Field(description="Third act of the chapter")

def get_completion(prompt):
    """Helper function to get completion from ChatOpenAI"""
    messages = [{"role": "user", "content": prompt}]
    response = model.invoke(messages)
    return response.content

def get_story_structure(story_prompt):
    """Determine the appropriate story structure based on the given prompt"""
    prompt = story_structure_chooser.format(story_prompt=story_prompt)
    return get_completion(prompt)

def get_story_structure_summerize(story_structure):
    """Summarize the determined story structure"""
    prompt = summarize_story_structure.format(story_structure=story_structure)
    return get_completion(prompt)

class PopularWorks(BaseModel):
    works: list[str] = Field(description="List of popular works mentioned")

def get_popular_works_json(structure):
    """Extract popular works mentioned in the structure and return as JSON"""
    parser = JsonOutputParser(pydantic_object=PopularWorks)
    
    prompt = PromptTemplate(
        template=popular_works_prompt + "\n{format_instructions}\n",
        input_variables=["structure"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    return chain.invoke({"structure": structure})["works"]

def get_story_blue_print(story_structure_summarized, story_prompt):
    """Generate a detailed story blueprint based on the summarized structure and original prompt"""
    prompt = blueprint_prompt.format(story_structure=story_structure_summarized, story_prompt=story_prompt)
    return get_completion(prompt)

def get_chapter_json(blueprint):
    """Create a JSON object of chapters based on the story blueprint"""
    parser = JsonOutputParser(pydantic_object=Chapters)
    
    prompt = PromptTemplate(
        template=chapter_json_prompt + "\n{format_instructions}\n",
        input_variables=["story_blueprint"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    return chain.invoke({"story_blueprint": blueprint})

def generate_acts(blueprint, chapter_number, chapter_title, chapter_desc):
    """Generate three acts for a given chapter"""
    prompt = act_generator_prompt.format(
        story_blue_print=blueprint,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        chapter_desc=chapter_desc
    )
    return get_completion(prompt)

def convert_acts_to_json(acts_plain_text):
    """Convert the generated acts into JSON format"""
    parser = JsonOutputParser(pydantic_object=Acts)
    prompt = PromptTemplate(
        template=acts_json + "\n{format_instructions}\n",
        input_variables=["acts"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    return chain.invoke({"acts": acts_plain_text})

def get_quotes_for_work(name: str) -> Optional[List[str]]:
    """
    Search for a work in Wikiquote and retrieve up to 10 quotes if found.
    
    Args:
    name (str): The name of the work to search for.
    
    Returns:
    Optional[List[str]]: A list of up to 10 quotes if found, None otherwise.
    """
    search_results = wikiquote.search(name)
    
    if not search_results:
        return None
    
    work_name = search_results[0]
    
    try:
        quotes = wikiquote.quotes(work_name, max_quotes=10)
        return quotes
    except Exception as e:
        print(f"Error retrieving quotes for '{work_name}': {e}")
        return None

def write_act(prompt, blueprint, chapter_desc, act_number, chapter_number, act_description, act_writing_advice, previous_text="", original_prompt="", snippets_from_famous_works="", previous_chapter_summaries=[], previous_acts_summary="", ending_lines=""):
    """Write the actual story text for a given act"""
    writing_advice = act_writing_advice if act_writing_advice != None else ""

    format_dict = {
        "story_blueprint": blueprint,
        "chapter_desc": chapter_desc,
        "act_number": act_number + 1,
        "chapter_number": chapter_number,
        "act_description": act_description,
        "act_writing_advice": writing_advice,
        "original_prompt": original_prompt,
        "famous_work_reference": snippets_from_famous_works,
        "chapters_summery": '\n\n'.join(previous_chapter_summaries)
    }
    
    # Only include previous_text if the placeholder exists in the prompt
    if "{previous_acts}" in prompt:
        format_dict["previous_acts"] = previous_acts_summary
        format_dict["lasting_lines"] = ending_lines

    log(f"[{datetime.now()}] format dict : {format_dict}")
    full_prompt = prompt.format(**format_dict)
    log(f"[{datetime.now()}] {full_prompt}")
    return get_completion(full_prompt)

def summarize(text):
    prompt = summarize_story_part.format(story_part=text)
    res = get_completion(prompt)
    log(f"[{datetime.now()}] summery result {res}")
    return res


log_data = ""

def log(log_text, use_streamlit=True):
    global log_data
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_line = f"{timestamp} {log_text}\n\n"
    log_data += log_line

    if USE_STREAMLIT and use_streamlit:
        log_area.markdown(f"\n{log_data}\n\n")
    else:
        print(log_line, end='')

def format_quotes(work_name, quotes):
    if quotes is None:
        return f"{work_name}:\nNo quotes found."
    return f"{work_name}:\n" + "\n".join(quotes)

def generate_story(prompt, inject_popular_work_reference=True, generate_story_structure_beforehand=True, maximum_chapters=None):
    # st.header("Story") if USE_STREAMLIT else print("Story")
    log("Starting story generation process")

    # Get the initial story prompt from the user
    story_prompt = prompt
    log(f"User provided story prompt: {story_prompt}")

    story = {}

    story_structure_summarized = None
    if generate_story_structure_beforehand:
        # Determine and summarize the story structure
        log("Determining story structure")
        story_structure = get_story_structure(story_prompt)
        log(f"Story structure determined: {story_structure}")

        log("Summarizing story structure")
        story_structure_summarized = get_story_structure_summerize(story_structure)
        log(f"Summarized story structure: {story_structure_summarized}")

    famous_work_reference = ""
    if inject_popular_work_reference:
        log("Determining popular works")
        similar_popular_works = get_popular_works_json(story_structure if generate_story_structure_beforehand else story_prompt)
        log(f"Similar popular works: {similar_popular_works}")

        famous_work_references = []
        for work in similar_popular_works:
            quotes = get_quotes_for_work(work)
            famous_work_references.append(format_quotes(work, quotes))

        famous_work_reference = "\n\n".join(famous_work_references)

    # Generate the story blueprint
    log("Generating story blueprint")
    blueprint = get_story_blue_print(story_structure_summarized, story_prompt)
    log(f"Story blueprint generated: {blueprint}")

    # Get the chapter JSON
    log("Generating chapter JSON")
    chapters = get_chapter_json(blueprint)["chapters"]
    log(f"Chapter JSON generated: {chapters}")

    # Generate the story content for each chapter and act
    
    # Before the main loop, add this:
    chapter_summaries = []

    for chapter_index, (chapter_title, chapter_desc) in enumerate(chapters.items()):
        chapter_number = chapter_index + 1
        log(f"Processing Chapter {chapter_number}: {chapter_title}")
        if USE_STREAMLIT:
            st.subheader(f"Chapter {chapter_number}: {chapter_title}")

        story[chapter_title] = ""

        # Generate and process acts for the current chapter
        log(f"Generating acts for Chapter {chapter_number}")
        acts_plain_text = generate_acts(blueprint, chapter_number, chapter_title, chapter_desc)
        log(f"Acts generated for Chapter {chapter_number}: {acts_plain_text}")

        log(f"Converting acts to JSON for Chapter {chapter_number}")
        acts = convert_acts_to_json(acts_plain_text)
        log(f"Acts converted to JSON for Chapter {chapter_number}: {acts}")

        # Write the story text for each act
        for act_index, (act_key, act_content) in enumerate(acts.items()):
            if maximum_chapters is not None and chapter_index >= maximum_chapters:
                break
            act_number = act_index + 1
            log(f"Writing Act {act_number} for Chapter {chapter_number}")
            
            act_prompt = write_act_prompt
            if act_index > 0:
                act_prompt += write_act_extra

            # Get previous acts summary
            previous_acts_summary = summarize(story[chapter_title]) if story[chapter_title] else ""

            # Get ending lines of current chapter
            ending_lines = story[chapter_title][-900:] if story[chapter_title] else ""

            act_text = write_act(
                act_prompt, 
                blueprint, 
                chapter_desc, 
                act_index, 
                chapter_number, 
                act_content["description"], 
                act_content.get("writingAdvice", None),
                original_prompt=prompt, 
                snippets_from_famous_works=famous_work_reference,
                previous_chapter_summaries=chapter_summaries,
                previous_acts_summary=previous_acts_summary,
                ending_lines=ending_lines
            )

            log(f"Act {act_number} written for Chapter {chapter_number}")
            log(f"Act {act_number} content for Chapter {chapter_number}: {act_text[:100]}...") # Print first 100 characters of the act
            if USE_STREAMLIT:
                st.write(act_text)
                st.text("")
            story[chapter_title] += "\n" + act_text
            log(act_text)

        # After finishing the chapter, summarize it and add to chapter_summaries
        chapter_summary = summarize(story[chapter_title])
        chapter_summaries.append(chapter_summary)

    # Combine all chapters into the full story text
    full_text = "\n\n".join(story.values())
    log("Story generation complete")
    log(f"Full story text (first 200 characters): {full_text[:200]}...")

    return full_text

if USE_STREAMLIT:
    # Custom CSS to reduce font size in the sidebar
    st.markdown("""
        <style>
        .small-font {
            font-size:0.8rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Streamlit UI setup
    st.title("Story Generator")
    story_prompt = st.text_input("Enter your story prompt:")
    generate_button = st.button("Generate Story")

    # Create a sidebar for logs
    sidebar = st.sidebar
    sidebar.title("Logs")
    log_area = sidebar.empty()

    if generate_button:
        if story_prompt:
            generate_story(story_prompt)
        else:
            st.warning("Please enter a story prompt!")
else:
    # If not running in Streamlit, you can add command-line interface here
    if len(sys.argv) > 1:
        story_prompt = sys.argv[1]
        generate_story(story_prompt)
    else:
        print("Please provide a story prompt as a command-line argument.")
