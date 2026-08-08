# User Guide (Non-Technical)

1. Open `notebooks/NewsBot_Intelligence_System_2.0.ipynb` in Google Colab.
2. Run all cells (Runtime → Run all). Wait a few minutes for training to finish.
3. To analyze your own article: edit the `new_article` variable in the "Live demo" cell and
   re-run it. You'll get: category, sentiment, key people/organizations/places, a short summary,
   and a one-paragraph brief.
4. To ask a question: use `system.ask("your question")`, e.g.
   `system.ask("how many sport articles are there")` or
   `system.ask("show me positive tech news")`.
5. To search: `system.search("your topic", top_k=5)` returns the most relevant articles even if
   they don't share exact keywords with your query.
