


### Steps: Navigate and Testing

a. Activate the .venv file in the backend folder .
   source backend/.venv/bin/activate (To create a .venv run python3.11 -m venv .venv)

b. Install the required libraries by running 
   pip install -r requirements.txt

   For Installing scikit-learn, we should have Python version below 3.12 or use 3.11
   To use Python 3.11 in the venv run " brew install python@3.11" and verify using "python --version"

c.Run " python3 -m backend.test_system "
  Gives the result of ML model with Accuracy and other metrics.

d. In the terminal include 
"export OPEN_API_KEY="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" OR "OPEN_API_KEY" .

e. Run " uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 "
 
   URL for the frontend will be displayed along with the model metrics .

f. Now open Terminal 2 and run
   " cd path_for_frontend_folder" 

g. Run " python3 -m http.server 5500 "
   
H. Open Index.html in the frontend folder and upload the test file and click on Process. 

   An ouput with multi cases and with confidence score will be generated.




