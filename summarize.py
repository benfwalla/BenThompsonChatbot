from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import OpenAI
from pprint import pprint
import dotenv

dotenv.load_dotenv()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]


def summarize_article(article_title, markdown_content):
    """Summarizes the content of a given markdown string using OpenAI's GPT-3.5 model and map-reduce approach"""
    if "* * *" in markdown_content:
        markdown_content = ''.join(markdown_content.split("* * *")[:-1])

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    section_splits = markdown_splitter.split_text(markdown_content)

    openai_client = OpenAI()

    article_type = "interview" if "Interview" in article_title else "article"

    section_summaries = []
    for i, section in enumerate(section_splits):
        section_content = section.page_content
        section_header = list(section.metadata.values())[-1]
        section_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Summarize the following snippet of a Stratechery {article_type} in markdown. "
                                              f"Use 'Ben' when describing the author and his points in the third person. Do not use descriptive clauses like 'in the interview,'. "
                                              f"Be concise and objective, with 3-5 sentences per section. Return a plain text paragraph, no formatting or new lines: {section_content}"}
            ]
        )
        section_summary = section_header + ": " + section_completion.choices[0].message.content
        section_summaries.append(section_summary)
        print(f"({i + 1}/{len(section_splits)}): {section_summary}")

    section_summaries = '\n'.join(section_summaries)
    article_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"The following is a set of summaries from a Stratechery article split by its sections: {section_summaries} "
                                          f"Take these and place it in a packaged, paragraph summary about the article. "
                                          f"In the event of an interview, intuit what the name abbreviations are from the section headers and use their names. "
                                          f"Mention the name of every section. The summary should use all the points mentioned below. "
                                          f"Return plain text paragraph, no formatting and no new lines."}
        ]
    )
    print("Full summary: " + article_completion.choices[0].message.content + "\n\n")
    return article_completion.choices[0].message.content
