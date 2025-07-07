## IMPORTANT NOTE:
Make sure all required libraries are downloaded for the dashboard libraries used:
- Pandas
- Numpy
- Streamlit
- blacksholes can be found here[https://pypi.org/project/blackscholes/]

### Run Command
```bash
streamlit run ./dashboard.py
```

Additional note the file path is relative to the linux file path method convention. If used in non-unix machines the path in the code needs to be changed for the (MODEL_PATH and SCALER_PATH).

The best-performing model is TabPFN. It is used to predict american_op - european_op. Also note, american_op is not used as a feature in X values and for the prediction it is using y = f(X) + european_op, where f(X) is the output of the model. This method produced the best performance in the testing set with an MRE of 0.308. Additionally, the prediction time includes transforming using the standard scaler, and including extra features (including calculating european_op). The model was trained on GPU.
