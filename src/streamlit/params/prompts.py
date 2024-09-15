
import json
from langchain.prompts.prompt import PromptTemplate

# Prompts utilisés pour la génération de résumés
# Ce sont ces deux constantes qui sont utilisées dans le code pour générer les résumés
short_prompt = PromptTemplate(
        template="""
        Please summarize the following text:
        {document}
        """,
        input_variables=["document"],
    )
elaborated_prompt_random = PromptTemplate(
        template="""
        Read the following text and give a short, clear summary of the main facts or ideas about it.
        Use markdown style and bulletpoints.

        Text to summarize:
        {document}
        """,
        input_variables=["document"],
    )

# Prompts utilisés pour l'évaluation des résumés
example_json={
    "Clarity":"(numeric value 1-5)",
    "Accuracy":"(numeric value 1-5)",
    "Coverage":"(numeric value 1-5)",
    "Overall quality":"(numeric value 1-5)",
    "Explanations": 
    {
        "Clarity": "Your explanations about clarity....",
        "Accuracy": "Your explanations about accuracy....",
        "Coverage": "Your explanations about coverage....",
        "Overall quality": "Your explanations about overall quality....",
    }}
example_json_string=json.dumps(example_json)

example_json_refBased={
    "Accuracy": "(numeric value 1-5)",
    "Conciseness": "(numeric value 1-5)",
    "Structure": "(numeric value 1-5)",
    "Explanations": {
        "Accuracy": "Your explanations about accuracy....",
        "Conciseness": "Your explanations about conciseness....",
        "Structure": "Your explanations about structure....",
    }
}
example_json_string_refBased=json.dumps(example_json_refBased)

evaluation_refFree_prompt = PromptTemplate(
    template=
        """
        You are an expert in evaluating summaries. 
        Assess the provided summary against its source document using the following four dimensions:

        (1) Clarity: whether the summary is reader-friendly and expresses ideas clearly.
        (2) Accuracy: whether the summary contains the same information as the source document.
        (3) Coverage: how well the summary covers the important information from the source document.
        (4) Overall quality: how good the summary overall at representing the source document; a good summary is a shorter piece of text that has the essence of the original, tries to convey the same information as the source document.

        For each dimension, give a numerical rating on a Likert-scale from 1 to 5 (1 = Poor, 5 = Excellent). Ensure that each of the first four keys contains only a single numeric value.
        Your response should be in JSON format with five keys:
        - 'Clarity': Numeric score (1-5) indicating the clarity of the summary.
        - 'Accuracy': Numeric score (1-5) indicating the accuracy of the summary.
        - 'Coverage': Numeric score (1-5) indicating the coverage of the summary.
        - 'Overall quality': Numeric score (1-5) indicating the overall quality of the summary.
        - 'Explanations': A brief explanation for each score given, detailing the reasons behind the evaluation.

        Materials for evaluation:
        - Summary: {summary}
        - Source Document: {document}

        Example output format:
        {example_output}
        """,
        input_variables=["document","summary"],
    )
evaluation_refBased_prompt = PromptTemplate(
    template=
        """
        You are an expert in comparing summaries. 
        Assess the provided summary against the reference summary based on the following three dimensions:

        (1) Accuracy: Evaluate how well the provided summary reflects the same information as the reference summary.
        (2) Conciseness: Assess whether the provided summary presents the information as concisely as the reference summary, avoiding unnecessary details and focusing on the core message.
        (3) Structure: Determine if the provided summary follows a similar structure to the reference summary.

        For each dimension, give a numerical rating on a Likert-scale from 1 to 5 (1 = Poor, 5 = Excellent). Ensure that each of the first three keys contains only a single numeric value.
        Your response should be in JSON format with four keys:
        - 'Accuracy': Numeric score (1-5) indicating the accuracy of the summary.
        - 'Conciseness': Numeric score (1-5) indicating the conciseness of the summary.
        - 'Structure': Numeric score (1-5) indicating how well the summary mirrors the structure of the reference.
        - 'Explanations': A brief explanation for each score given, detailing the reasons behind the evaluation.

        Materials for evaluation:
        - Summary to Analyze: {summary}
        - Reference Summary: {ref_summary}

        Example output format:
        {example_output}
        """,
        input_variables=["ref_summary","summary"],
    )

# C'est ce code qui est utilisé pour générer les prompts d'évaluation des résumés
evaluation_prompts={
    "refFree":
    {
        "prompt_standard":evaluation_refFree_prompt,
        "example_output":example_json_string
    },
    "refBased":
    {
        "prompt_standard":evaluation_refBased_prompt,
        "example_output":example_json_string_refBased
    }
}