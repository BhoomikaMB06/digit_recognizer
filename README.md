## Run the web app

```
pip install -r requirements.txt
python digit_recognizer1.py
python app.py
```

1. `pip install -r requirements.txt` installs the libraries.
2. `python digit_recognizer1.py` trains the model (lr = 0.001) and saves `digit_model_lr0.001.pth`.
3. `python app.py` starts the server. Open http://127.0.0.1:5000 and draw a digit.