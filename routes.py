"""REST API for saying hi."""
# from app import app as application
from flask import *
# import send_static_file
import os
import datetime
import jsonify
app = Flask(__name__)
BASE_URL = os.path.abspath(os.path.dirname(__file__))
CLIENT_APP_FOLDER = os.path.join(BASE_URL, "dist")
jinja_options = app.jinja_options.copy()

jinja_options.update(dict(
    block_start_string='<%',
    block_end_string='%>',
    variable_start_string='%%',
    variable_end_string='%%',
    comment_start_string='<#',
    comment_end_string='#>'
))
app.jinja_options = jinja_options
import numpy as np
import datetime
np.random.seed(42)

rooms = {}

@app.route('/input', methods=["POST"])
def addToRoom():
	print(request.form)
	r = request.get_json()
	room = r['room']
	score = r['score']
	if check(room):
		return json.dumps(compare(score, rooms[room]))
	rooms[room] = score
	return json.dumps("nobody else in the room!")

def check(roomNumber):
	return roomNumber in rooms

def compare(one, two):
	print(request.form)
	wellford = one
	panda = two
	return wellford > panda

@app.route('/check', methods=["GET"])
def apiCheck():
	return json.dumps(check(request.args["room"]))

if __name__ == "__main__":
	# Setting debug to True enables debug output. This line should be
	# removed before deploying a production app.
	app.debug = True
	app.run()