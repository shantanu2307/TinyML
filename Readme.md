Objective: To create a low powered deep learning model to detect paper, plastic, trash, metal, cardboard and glass for smart dustbins in order to segregate waste into bio-degradable and non bio-degradable category.

Dataset Used: Trashnet (Kaggle)

Procedure: We trained an InceptionResnetV2 model on the trashnet dataset, which has 6 classes and 2527 images. We got an accuracy around 85%. Then we extracted the features from the last layer of the model and inserted those features to the CNN-XGBOOST model. The final accuracy that we got was 94.2%. Then we used TFlite to minify our deep learning model and with the help of streamlit cloud, we deployed it on the internet.

Accuracy: 94.2%

Tools: Streamlit-Cloud

Libraries Used: TFlite, Tensorflow, Sklearn, numpy, pandas, streamlit
