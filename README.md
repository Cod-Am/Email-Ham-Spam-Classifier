Email Ham-Spam Classifier:
The model is able to classify emails into Spam or Ham over a precision of 1.0 and an accuracy of 0.987. The algorithm used for the model is SGD Classifier.
The dataset contained 5572 records and data preprocessing was performed using the NLTK library. The dataset was first word tokenized, then converted to lowercase. 
After this stop words were removed and then each word was stemmed. The resulting corpus was then tfidf vectorized and then model training was done.
The project is currently on-going and I plan to deploy the model in future.
![image](https://github.com/user-attachments/assets/0b18c08a-4897-4fc4-bb58-4a4576c984f1)
