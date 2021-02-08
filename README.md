# Local Interpretable Model-agnostic Explanations

This project contains two Jupyter notebooks, which intend to give a hand-on introduction to explainable AI with LIME. 

![](data/header.jpg)

LIME is a novel technique allowing more insights into black box models like Deep Neural network, Support Vector Machines or Random Forrest. The working idea is surprisingly simple but turns out to work very fine, if we choose to observe the targeted model as a blackbox and only wish to interpret it's decision.

Following are cases where LIME shines:
- Explaining any classifier or regressor's output in a faithful and insightful way.

Where LIME struggles to prove its usefulness:
- Interpreting internal parameters of model (hidden layers of a neural network for example) and which roles they play for the decision. 

In order to achieve its desired property, LIME approximate the targeted model's decisions locally with a simple and explainable by design model, of which following belongs:
- Linear Regression
- Logistic Regression
- Decision Tree

By obtaining this simple model, humans behind the scene get a clearer idea of which input features are important for the final decision. Ethically we naturally want to eliminate biases in data and avoid sensible data to play a greate role such as:
- Gender
- Race
- Religious or political orientation

### Interpreting Random Forrest Classifier (RFC)

In the notebook `Explain Random Forrest.ipynb` we examine the [https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data](German Credit) dataset. Application for credits can sometimes be someone's chance to start a new business, getting into higher education or a life-saving medical therapy. Of course, we do not wish to leave the final decision to a discriminating model and hence desire to eliminate bias. 

In this example, we trained a RFC with 50 sub-estimators and verified the final result with cross validation (0.95 ± 0.002). The final result may sound good on paper, but with LIME we found a very fundemental problem with the model.

![](data/explain_random_forrest.png)

It turns out the model's final decision depent greatly on the feature `SK_ID_CURR` (customer's unique ID). The greater/new the ID, the lower chance the applicant has to get the credit (poor millenials). By removing this feature from the training dataset, the model's accuracy sinks back to (0.64 ± 0.003), which is only slightly better than mere chance. This is by definition a discriminating model, which we do not wish to deploy into production.

But there are many other interesting features in this plot. By `FLAG_OWN_CAR` we can conclude that the chance is higher for car owner to obtain the credit, but only if your car is not too old (`OWN_CAR_AGE`).
