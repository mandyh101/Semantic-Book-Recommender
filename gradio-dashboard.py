import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
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
        query: str,
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
    book_recs = books[books["isbn13"].isin(books_list).head(initial_top_k)]

    #4 create a category filter
    ## if category does not equal all, filter the books by the simple_categories values and limit to the final_top_k (16)
    ## otherwise show all books matching the similarity search but limit to the final_top_k (16)
    if category != "All":
        books_recs = book_recs[book_recs["simple_categories"] == category][final_top_k]
    else:
        books_recs = book_recs[final_top_k]

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


