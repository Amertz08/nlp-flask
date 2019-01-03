import pickle
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from sklearn.externals import joblib


app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config["SECRET_KEY"] = "hello"


class SpamForm(FlaskForm):
    message = TextAreaField("Message", validators=[DataRequired()])
    submit = SubmitField("Check if spam")


@app.route("/")
def home():
    form = SpamForm()
    return render_template("home.html", form=form)


@app.route("/predict", methods=["POST"])
def predict():
    form = SpamForm()
    if form.validate_on_submit():
        # Load our persistent CountVectorizer
        with open("cbobj.pkl", "rb") as cv_file:
            cv = pickle.load(cv_file)

        # Load our persistent NLP model
        with open("NB_spam_model.pkl", "rb") as model:
            clf = joblib.load(model)

        # Make prediction
        data = [form.message.data]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template("result.html", prediction=my_prediction)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
