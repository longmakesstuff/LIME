# Local Interpretable Model-agnostic Explanations

This project contains two Jupyter notebooks, which intend to give a hand-on introduction to explainable AI with LIME. 

LIME is a novel technique allowing more insights into black box models like Deep Neural network, Support Vector Machines or Random Forrest. The working idea is surprisingly simple but turns out to work very fine, if we choose to observe the targeted model as a blackbox and only wish to interpret it's decision.

Following are cases where LIME shines:
- Explaining any classifier or regressor's output in a faithful and insightful way.

Where LIME struggles:
- Interpreting internal parameters of model (hidden layers of a neural network for example) and which roles they play for the decision. 

In order to achieve this property, LIME approximate the targeted model's decisions locally with a simple and explainable by design model, of which following belongs:
- Linear Regression
- Logistic Regression
- Decision Tree

By obtaining this simple model, humans behind the scene get a clearer idea of which input features are important for the final decision. Ethically we naturally want to eliminate biases in data and avoid sensible data to play a greate role such as:
- Gender
- Race
- Religious or political orientation

![](data/header.jpg)

