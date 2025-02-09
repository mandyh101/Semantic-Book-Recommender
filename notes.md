## Packages
1. kagglehub - data access
2. matplotlib - data visualisation dependency or seaborn
3. seaborn - statistical data visualisation
4. pandas - data science analysis tool
5. python-dotenv - for working with openAPI
5. langchain - a collection of languages for working with LLMs
- langchain-community: for implementing base interfaces in Langchain core packages
- langchain-openai - for working with open ai
- langchain--chroma - for working with vector databases
6. transformers - hugging face package for working with LLMs
7. gradio - for building a demo or web app to easily share an MML app: https://www.gradio.app/guides/quickstart
8. jupyter notebook - notebook env for interactive computing
9. ipywidgets - widgets

## Components required for this book recommender
1. A vector database that allows us to find the most similar books to a query.
2. Use text classification (zero-shot) to sort books into fiction or non-fiction so users can also filter books based on this category.
3. We found out how likely each book description is to have a certain emotional tone using fine-tuning text classification (so we can apply emotional semantics to search).
4. A user-friendly interface to present the book recommender in a dashboard that people can use to get their book recommendations.

## Next steps
1. deploy gradio dashboard with hugging face
2. write up learnings
3. publish to portfolio