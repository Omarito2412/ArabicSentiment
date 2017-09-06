## Arabic Sentiment Analysis
### I've made this repository as part of an interview task
### This repo contains a few scripts to calculate sentiment of a given tweet based on the [ASTD](https://github.com/mahmoudnabil/ASTD)

I've mainly used IBM watson to train the classifier (Trial period from IBM Bluemix)
To improve on Watson I've also trained a few other models which are:
1. A Bidirectional LSTM model
2. A Gradient boosted machine
3. A Naive bayes model

In the end I use the predictions from all of these models and average the probabilities to achieve higher accuracies

I have also used the [NileULex](https://github.com/NileTMRG/NileULex).

Here's a table demonstrating the results.
(I don't use the lexicon in any of the ensembles)

| Model               | Score         
| -------------       |:-------------:
| Lexicon only        | %41.3
| IBM Watson          | %68.1
| LightGBM            | %68.4
| Naive Bayes         | %67.3
| Bi-LSTM             | %63.4
| Ensemble(All)       | %69.16
| (Watson + LGBM + NB)| %69.33
| (Watson + LGBM)     | **%69.5**


> You can rerun the script to recreate the results by running the run_all.sh script


