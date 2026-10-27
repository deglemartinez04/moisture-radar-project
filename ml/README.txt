ML CHECK  -  run the machine learning checks by double-clicking
===============================================================

All the data in these checks is SIMULATED. They show that the code works and that the
~80% result is reproducible. They do not show how well the real radar will do.

FIRST TIME ONLY
  Double-click   1_SETUP_once.bat
  (needs internet; about 2-5 minutes; installs Python packages outside this folder)

THEN DOUBLE-CLICK ANY OF THESE
  2_RUN_CHECK.bat              Simulates 30 recordings, builds the training table,
                               trains the model and prints the accuracy.
  3_RETRAIN_SAVED_DATASET.bat  Re-trains on the dataset that is already saved here.
                               Fastest (about 10 seconds).
  4_FIVE_RUNS.bat              Five different simulated datasets, to see how much the
                               result varies (about 1 minute).
  5_SHUFFLE_TEST.bat           Shuffles the labels. Accuracy should fall to chance
                               (about 33%). Proves the test is not cheating.

EACH ONE
  - shows the output in the window and waits for a key press at the end
  - saves the output in the "results" folder, so you can open it later

WHAT YOU SHOULD SEE
  2 and 3:  Window accuracy 79.9%, Recording accuracy 80.0% (24/30), Chance level 33.3%
  4:        recording accuracy 80.0%, 73.3%, 90.0%, 76.7%, 93.3%
  5:        real labels ~80%, shuffled labels ~27%

THE LINES THAT MATTER IN THE REPORT
  Window accuracy      how often one 10-frame reading is right
  Recording accuracy   whether the majority vote over a whole recording is right
  Chance level         what random guessing would score (33.3% for 3 classes)
  Confusion matrix     rows = true class, columns = predicted class
  Misclassified        the recordings it got wrong (all near the 12% / 20% cutoffs)

FOLDER CONTENTS
  ml\                  the machine learning code
  saved_dataset\       the dataset used for the 80% result
  results\             created when you run a check
  shuffle_labels.py    helper for the shuffle test

IF SOMETHING GOES WRONG
  "Not set up yet"     run 1_SETUP_once.bat first
  Python not found     install Python 3 from python.org (tick "Add python.exe to PATH")
