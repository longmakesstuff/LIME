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

By obtaining this simple model, humans behind the scene has a better chance of explaining a very complex model's decisions, hence getting the best of both worlds: explainability of traditional simple models and state of the art result of modern black box models. Ethically we naturally want to eliminate biases in data and avoid sensible data to play a greate role such as:
- Gender
- Race
- Religious or political orientation

### Interpreting Random Forrest Classifier (RFC)

In the notebook `Explain Random Forrest.ipynb` we examine the [https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data](German Credit) dataset. Application for credits can sometimes be someone's chance to start a new business, getting into higher education or a life-saving medical therapy. Of course, we do not wish to leave the final decision to a discriminating model and hence desire to eliminate bias. 

In this example, we trained a RFC with 50 sub-estimators and verified the final result with cross validation (0.95 ± 0.002). The final result may sound good on paper, but with LIME we found a very fundemental problem with the model.

![](data/explain_random_forrest.png)

It turns out the model's final decision depent greatly on the feature `SK_ID_CURR` (customer's unique ID). The greater/new the ID, the lower chance the applicant has to get the credit (poor millenials). By removing this feature from the training dataset, the model's accuracy sinks back to (0.64 ± 0.003), which is only slightly better than mere chance. This is by definition a discriminating model, which we do not wish to deploy into production.

But there are many other interesting features in this plot. By `FLAG_OWN_CAR` we can conclude that the chance is higher for car owner to obtain the credit, but only if your car is not too old (`OWN_CAR_AGE`).

To verify the interpretation is not mere by chance, we test 100 outputs of LIME with variance analysis and to sort out the hypothesis that the result is a product of pure randomness.

### Interpreting Convolution Neural Network (CNN)

In this example, we train a deep CNN to help us to decide if there is a cat on an image. There is a funny anecdote, where a military R&D trained a network to detect tank on images. The model worked in the training phase very well so it was deployed into test environment, where its outputs suddenly became useless. It turns out that all the training data with a tank had an unintentional water mark, which made the final model worthless. 

To avoid this kind of problem, I deployed LIME to point out important features of the original pictures for model's decision. 

![](data/explain_neural_network.png)

The brighter the pixels, the more weight it contribute to the classification `cat`. As we can see in the pictures, the orange's cat pixels seem to be the key for the detection. While the dog's pixels rather told the model "Hey, I am clearly not a cat.".


## References

- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
