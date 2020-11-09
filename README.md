# Flask Heroku Example
_(This repo is part of our [Free Flask Tutorial](https://flask-tutorial.com))_

This repo contains some sample code to deploy a simple (but complete) Flask application to [Heroku](https://heroku.com). The deployed app counts with the following features:

* Running Python 3.6 🐍
* Access to a Postgres Database 📘
* Static Files management with [WhiteNoise](http://whitenoise.evans.io/en/stable/) 🔌

**There's a detailed video lesson on how to perform the deploy in our [Free Flask Tutorial](https://flask-tutorial.com).**

## Summary of steps to deploy your app
_(Assuming you've already created an account with Heroku)_

##### 1. Clone the repo
```bash
$  cd Flask_sample
```

##### 2. Login to Heroku
```bash
$ heroku login
```

##### 3. Create your Heroku apps
```bash
$ heroku create
```

##### 4. Set the Python Path
```bash
$ heroku config:set PYTHONPATH=Flask_sample





##### 5. Deploy & Profit
```bash
$ git push heroku master
```

## Running the app locally
_(You need to have installed Postgres locally to run the app. For a simpler sqlite alternative, please check the aforementioned tutorial)_

```bash

# Install dependencies
$ pip install -r requirements.txt
# Run the app
$ python Flask_sample/app.py
# Now point your browser to localhost:5000
```