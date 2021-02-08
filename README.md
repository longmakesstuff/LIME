# Local Interpretable Model-agnostic Explanations

This project gives a short and practical hand-on introduction to explainable AI with LIME. 

![](data/header.jpg)

LIME is a novel technique allowing deeper insights into black box models's decision making. The working idea is simple but turns out to work surprisingly fine, assuming we only wish to interpret the blackbox's decisions and not its topology.

Following are use cases where LIME shines:
- Explaining any classifier or regressor's output in a faithful way.

Where LIME struggles to prove its usefulness:
- Interpreting internal parameters of model (hidden layers of a neural network for example) and which roles they play for the decision. 

In order to achieve its desired property, LIME approximate the targeted model's decisions locally with a simple and explainable by design model, for example:
- Linear Regression
- Logistic Regression

Those kinds of model suffer from less generalizability, but they are explainable by nature. Linear and logistic regression for example tell you directly if an input parameter contributes positively or negatively to the end result. 

On the other side of the spectrum we have Neural Network, Support Vector Machines or Random Forrest. These shiny models can yield excellent results and were proven to be very robust, but because of their complexity, it is almost impossible to know how they come to their decisions, so they have plenty difficulties to gain trust of wider audience.

By using LIME for model validation, we can get the best of both worlds: explainability of simple but weak models and state of the art results from modern black box models. Furthermore, from the ethical standpoint, we naturally want to eliminate biases in data and avoid sensible data to play a greate role such as:
- Gender.
- Race.
- Religious or political orientation.

### Interpreting Random Forrest Classifier (RFC)

In the notebook `Explain Random Forrest.ipynb` we examine the [German Credit](https://www.kaggle.com/uciml/german-credit) dataset. Application for credits can sometimes be someone's chance to start a new business, getting into higher education or a life-saving medical therapy. Of course, we do not wish to leave the final decision to a discriminating model and hence desire to eliminate bias. 

In this example, we trained a RFC with 50 sub-estimators and verified the final result with cross validation (0.95 ± 0.002). The model's performance may sound good on paper, but with help of LIME we found a very fundemental problem with the model (Hints for upcoming illustration: the higher a feature's value, the less likely your application will get approved).

![](data/explain_random_forrest.png)

It turns out the model's final decision depent greatly on the feature `SK_ID_CURR` (customer's unique ID). The greater/newer the ID, the lower chance the applicant has to get the credit (new customers will have to take time to prove themselves worthy). By removing this feature from the training dataset, the model's accuracy sinks back to (0.64 ± 0.003), which is only slightly better than mere chance. This is by definition a discriminating model, which we do not wish to deploy into production.

But there are many other interesting features in this plot. By `FLAG_OWN_CAR` we can conclude that the chance is higher for car owner to obtain the credit, but only if your car is not too old (`OWN_CAR_AGE`).

To verify the interpretation is not mere by chance, we test 100 outputs of LIME with one-way ANOVA and to sort out the hypothesis that the result is a product of pure randomness.

### Interpreting Convolution Neural Network (CNN)

In this example, we train a deep CNN to help us to decide if there is a cat on an image. There is a funny anecdote, where a military R&D trained a network to detect tank on images. The model worked in the training phase very well, so it was deployed into test environment, where its outputs suddenly became useless. After weeks of debugging, the scientists of the project found out that all the training data with a tank had an unintentional water mark, which correlates highly with the final outputs, therefore making the model worthless in real life. 

To avoid this kind of problem, I deployed LIME to point out important features of the original pictures for model's decision. Beside classical cross validation, LIME shows use definitively which pixels were involved heavily in the inference and could draw our attention to possible biasnesses.  

![](data/explain_neural_network.png)

In the two inferences shown above, the brighter the pixels contribute more weight to the end-classification `is cat`. As we can see in the pictures, the orange's cat brighter pixels seem to be the key for the detection and the dog's darker pixels rather scream "Hey, I am clearly not a cat.".


## References

- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
