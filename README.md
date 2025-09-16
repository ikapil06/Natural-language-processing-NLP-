# Flask Application: Product Review Sentiment Analysis

This folder contains a simple Flask web application that performs
sentiment analysis on e‑commerce product reviews.  The application
implements a lexicon‑based approach, which classifies each review
into positive, negative or neutral categories by comparing the
preprocessed tokens against lists of positive and negative words.  In
keeping with common definitions of sentiment analysis, the goal is to
identify whether the emotional tone of a text is positive, negative or
neutral【127290445729235†L117-L120】.  Lexicon‑based methods work by
applying predefined rules and heuristics【127290445729235†L141-L146】.

## Files

* **app.py** – main Flask application.  It loads and preprocesses
  the dataset, computes aggregated statistics and exposes a single
  route.  Users can submit a review via a form and receive a
  sentiment prediction.  The page also displays a summary table of
  sentiment counts per product and a bar chart of overall
  distribution.
* **templates/index.html** – HTML template used to render the
  application’s user interface.  It includes a textarea for input,
  displays the prediction, the summary table and the bar chart.
* **requirements.txt** – list of Python packages required to run the
  application locally.
* **data/reviews.csv** – synthetic dataset of product reviews.  Each
  record has two columns: `product` and `review`.

## Running locally

1. Create and activate a Python virtual environment.
2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

   Note: `spacy` is optional.  If installed and the model
   `en_core_web_sm` is available, the application will lemmatize tokens
   during preprocessing; otherwise it will fall back to a simpler
   regex‑based tokenizer.

3. Start the development server:

   ```bash
   python app.py
   ```

4. Visit `http://localhost:5000` in your browser.  Enter a review in
   the textarea and click **Analyze** to see the predicted sentiment.
   The page also shows a table with counts of positive, negative and
   neutral reviews per product and a bar chart summarising the overall
   distribution.

## Extending the application

This demonstration uses a small synthetic lexicon.  For real‑world
applications you can expand the positive and negative word lists or
integrate a more sophisticated analyser such as NLTK’s VADER
sentiment scorer or spaCy’s built‑in sentiment models.  You can also
add routes to serve JSON responses or integrate the analysis into a
larger system.