from flask import Flask
from flask import request, current_app, abort
from functools import wraps
from flask.json import jsonify


app = Flask(__name__)
app.config.from_object('settings')


def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-TOKEN', None) != current_app.config['API_TOKEN']:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


@app.route('/predict', methods=['POST'])
@token_auth
def predict():
    from engines import content_engine
    item = request.get_data('item')
    num_predictions = 10
    if not item:
        return []
    return content_engine.predict(str(item), num_predictions, "id_desc_price.xlsx")


"""
curl -X GET -H "X-API-TOKEN: FOOBAR1" -H "Content-Type: application/json; charset=utf-8" http://127.0.0.1:5000/train
"""

@app.route('/train')
@token_auth
def train():
    from engines import content_engine
    #data_url = request.get_data('data-url', None)
    content_engine.train("id_desc_price.xlsx")
    return "\n\nSuccess: Model trained\n\n"


if __name__ == '__main__':
    app.debug = True
    app.run()
