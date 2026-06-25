from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory("css", filename)

@app.route("/js/<path:filename>")
def js(filename):
    return send_from_directory("js", filename)

@app.route("/img/<path:filename>")
def img(filename):
    return send_from_directory("img", filename)

@app.route("/fonts/<path:filename>")
def fonts(filename):
    return send_from_directory("fonts", filename)


if __name__ == "__main__":
    app.run(debug=True)