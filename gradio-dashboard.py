import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions_2.csv")
# first we want to use the thumbnail as a nice visual preview. We need to set the largest resolution to ensure consistent sizes and resolution by ensuring we get the largest size possible back from Google books using the string argument at teh end of the line below:
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# next, because a number of books don't have covers, we set an interim/default cover if a google book image preview does not exist
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#now we build the core functionality for semantic recommendations with our vector db
## read the tagged descriptions into the text load
raw_documents = TextLoader("tagged_description.txt").load()
## instantiate a character text splitter by new line
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
## apply the text splitter to our documents, with each chunk its own document
documents = text_splitter.split_documents(raw_documents)
## convert those documents into document embeddings using openAI embeddings and store them in a Chroma vector db
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# then create a function to retrieve the semantic recommendations from our books data, and apply filtering based on category and sorting based on emotional tone
def retrieve_semantic_recommendations(
        query: str = None,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # 1. run the similarity search over the vector db to get top results based on query string
    recs = db_books.similarity_search(query, k=initial_top_k)
    #2 get a list of ISBNs for all the returned recommendations
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    #3 now limit our data so we only show books with an isbn that matches an isbn in books_list
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    #4 create a category filter
    ## if category does not equal all, filter the books by the simple_categories values and limit to the final_top_k (16)
    ## otherwise show all books matching the similarity search but limit to the final_top_k (16)
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #5. then add a filter by emotion to sort results based on sematic probability
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    #6 finally return the top 16 book recommendations
    return book_recs

# Function to determine what is displayed on the gradio dashboard
def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    #1. get recommendations
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    #2. initialise empty results list to add our transformed data to
    results = []
    #3 loop through recommendations and transform data for results display
    for _, row in recommendations.iterrows():
        # 3.a. truncate description for preview display
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        # 3.b. display the authors in a list
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split)[:-1]}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        #4 combine book info into a caption string for display
        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        #5 append each result to the results list
        results.append((row["large_thumbnail"], caption))
    # return results to display
    return results

# DASHBOARD SET UP
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g. a story about friendship")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns=8, rows=2)

    # tell gradio that when a use clicks the submit button, call the recommended books fn, pass in the user query, selected category and tone if available) and return output
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output],
    )

    if __name__ == "__main__":
        dashboard.launch(share=True)

