from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)


def check_auth(username, password):
    return username == 'testuser' and password == 'testpassword'


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({'error': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/hello')
def hello():
    return 'Hello!'


@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify(data)


@app.route('/api/v1/users')
@requires_auth
def api_users():
    return jsonify({'users': ['user1', 'user2', 'user3']})


if __name__ == '__main__':
    app.run(debug=True)

