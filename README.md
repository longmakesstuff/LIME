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

In the notebook `Explain Random Forrest.ipynb` we examine the [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) dataset. Application for credits can sometimes be someone's chance to start a business or getting into higher education. Of course, we do not wish to leave the final decision to a discriminating model and hence desire to eliminate bias. 

In this example, we trained a RFC with 50 sub-estimators and verified the final result with cross validation (0.95 ± 0.002). The final result may sound good on paper, but with LIME we found a very fundemental problem with the model.

![](data/explain_random_forrest.png)
