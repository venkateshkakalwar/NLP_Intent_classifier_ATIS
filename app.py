from flask import Flask, request,render_template
import pickle

app = Flask(__name__)

cl = pickle.load(open('model.pkl','rb'))
tv = pickle.load(open('cv_transfomr.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tv.transform(data).toarray()
        my_prediction = str(cl.predict(vect)[0])
        return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
    
