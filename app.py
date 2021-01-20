from flask import Flask, render_template, session, redirect, url_for, request
from flask_session import Session
import math
import os
import sys

import utils.Filters as filter
import utils.Operations as op
app = Flask(__name__)
app.secret_key = "A0Zr98j"
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
Session(app)


@app.route("/")
def home():
    if "image" in session:
        show_image = "data:image/png;base64,"+session["image"]
        if "output" in session:
            output = session["output"]
            op_name = session["op_name"]
            return render_template("index.html", orginal=show_image, output=output, op_name=op_name)
        return render_template("index.html", orginal=show_image)
    return render_template("index.html", orginal="no Image")


@app.route("/setImage", methods=['POST', 'GET'])
def setImage():
    if request.method == "GET":
        return "<h1> it is a post method</h1>"
    image_mat = request.form["image"]
    session["image"] = image_mat
    if "output" in session:
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("home"))


@app.route("/save", methods=['GET'])
def saveChanges():
    if "output" in session:
        session["image"] = session["output"].replace(
            "data:image/png;base64,", "")
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("home"))


@app.route("/grayish", methods=['GET'])
def grayish():
    session["output"] = "data:image/png;base64," + \
        filter.grayish(session["image"])
    session["op_name"] = "GRAYISH"
    return redirect(url_for("home"))


@app.route("/pixel", methods=['GET'])
def pixelArt():
    session["output"] = "data:image/png;base64," + \
        filter.pixelArt(session["image"])
    session["op_name"] = "PIXXEL"
    return redirect(url_for("home"))


@app.route("/pop", methods=['GET'])
def popArt():
    session["output"] = "data:image/png;base64," + \
        filter.popArt(session["image"])
    session["op_name"] = "BONCUK"
    return redirect(url_for("home"))


@app.route("/inverse", methods=['GET'])
def inverse():
    session["output"] = "data:image/png;base64," + \
        op.inverse(session["image"])
    session["op_name"] = "INVERSE"
    return redirect(url_for("home"))


@app.route("/emboss", methods=['GET'])
def emboss():
    session["output"] = "data:image/png;base64," + \
        filter.emboss(session["image"])
    session["op_name"] = "EMBOSS"
    return redirect(url_for("home"))


@app.route("/mirror", methods=['GET'])
def mirror():
    session["output"] = "data:image/png;base64," + \
        op.mirror(session["image"])
    session["op_name"] = "MIRROR"
    return redirect(url_for("home"))


@app.route("/rotate", methods=['GET'])
def rotate():
    angle = int(request.headers["angle"])
    session["output"] = "data:image/png;base64," + \
        op.rotate(session["image"], angle)

    session["op_name"] = "ROTATE"
    return redirect(url_for("home"))


@app.route("/lumos", methods=['GET'])
def lumos():
    lumen = int(request.headers["lumen"])
    session["output"] = "data:image/png;base64," + \
        op.lumos(session["image"], lumen)

    session["op_name"] = "BRIGHT"
    return redirect(url_for("home"))


@app.route("/contrast", methods=['GET'])
def contrast():
    contrast = int(request.headers["contrast"])
    session["output"] = "data:image/png;base64," + \
        op.contrast(session["image"], contrast)

    session["op_name"] = "CONTRAST"
    return redirect(url_for("home"))


@app.route("/cropselect", methods=['GET'])
def crop_select():

    session["op_name"] = "CROP_SELECT"
    session["output"] = "select for crop"

    return redirect(url_for("home"))


@app.route("/crop", methods=['GET'])
def crop():

    p = request.headers["points"]
    w = int(request.headers["width"])
    p = p.split(',')
    points = []
    i = 0

    while i <= len(p)-2:
        points.append([math.floor(float(p[i])), math.floor(float(p[i+1]))])
        i += 2

    session["output"] = "data:image/png;base64," + \
        op.crop(session["image"], points, w)
    session["op_name"] = "CROP"

    return redirect(url_for("home"))


@app.route("/flip", methods=['GET'])
def flip():
    try:
        _hor = request.headers["hor"]
        if(_hor == "false"):
            hor = False
        else:
            hor = True
        _ver = request.headers["ver"]

        if(_ver == "false"):
            ver = False
        else:
            ver = True

    except:
        hor = False
        ver = False

    session["output"] = "data:image/png;base64," + \
        op.flip(session["image"], hor, ver)

    session["op_name"] = "FLIPPER"
    return redirect(url_for("home"))


@app.route("/oldtv", methods=['GET'])
def oldtv():
    session["output"] = "data:image/png;base64," + \
        filter.oldtv(session["image"])
    session["op_name"] = "90's TV"
    return redirect(url_for("home"))


@app.route("/sketch", methods=['GET'])
def sketch():
    session["output"] = "data:image/png;base64," + \
        filter.sketch(session["image"])
    session["op_name"] = "SKETCH"
    return redirect(url_for("home"))


@app.route("/splash", methods=['GET'])
def splash():
    session["output"] = "data:image/png;base64," + \
        filter.splash(session["image"])
    session["op_name"] = "SPLASH"
    return redirect(url_for("home"))


@app.route("/sepya", methods=['GET'])
def sepya():
    session["output"] = "data:image/png;base64," + \
        filter.sepya(session["image"])
    session["op_name"] = "SEPIA"
    return redirect(url_for("home"))


@app.route("/cartoon", methods=['GET'])
def cartoon():
    session["output"] = "data:image/png;base64," + \
        filter.cartoon(session["image"])
    session["op_name"] = "CARTOON"
    return redirect(url_for("home"))


@app.route("/oily", methods=['GET'])
def oily():
    session["output"] = "data:image/png;base64," + \
        filter.oily(session["image"])
    session["op_name"] = "OILY"
    return redirect(url_for("home"))


@app.route("/autocon", methods=['GET'])
def autocon():
    session["output"] = "data:image/png;base64," + \
        op.histogramEqualizer(session["image"])
    session["op_name"] = "EQUALIZED"
    return redirect(url_for("home"))


@app.route("/abstractify", methods=['GET'])
def abstractify():
    session["output"] = "data:image/png;base64," + \
        filter.abstractify(session["image"])
    session["op_name"] = "NOTIONAL"
    return redirect(url_for("home"))


@app.route("/balmy", methods=['GET'])
def balmy():
    session["output"] = "data:image/png;base64," + \
        filter.warm(session["image"])
    session["op_name"] = "BALMY"
    return redirect(url_for("home"))


@app.route("/cold", methods=['GET'])
def cold():
    session["output"] = "data:image/png;base64," + \
        filter.cold(session["image"])
    session["op_name"] = "FROSTBITE"
    return redirect(url_for("home"))


@app.route("/lines", methods=['GET'])
def lines():
    session["output"] = "data:image/png;base64," + \
        filter.lines(session["image"])
    session["op_name"] = "LINES"
    return redirect(url_for("home"))


@app.route("/blush", methods=['GET'])
def blush():
    session["output"] = "data:image/png;base64," + \
        filter.blush(session["image"])
    session["op_name"] = "BLUSH"
    return redirect(url_for("home"))


@app.route("/glass", methods=['GET'])
def glass():
    session["output"] = "data:image/png;base64," + \
        filter.glass(session["image"])
    session["op_name"] = "GLASS"
    return redirect(url_for("home"))


@app.route("/xpro", methods=['GET'])
def xpro():
    session["output"] = "data:image/png;base64," + \
        filter.xpro(session["image"])
    session["op_name"] = "XPRO"
    return redirect(url_for("home"))


@app.route("/daylight", methods=['GET'])
def daylight():
    session["output"] = "data:image/png;base64," + \
        filter.daylight(session["image"])
    session["op_name"] = "DAYLIGHT"
    return redirect(url_for("home"))


@app.route("/moon", methods=['GET'])
def moon():
    session["output"] = "data:image/png;base64," + \
        filter.moon(session["image"])
    session["op_name"] = "MOON"
    return redirect(url_for("home"))


@app.route("/blueish", methods=['GET'])
def blueish():
    session["output"] = "data:image/png;base64," + \
        filter.blueish(session["image"])
    session["op_name"] = "BLUEISH"
    return redirect(url_for("home"))


@app.route("/clear")
def clear():
    [session.pop(key) for key in list(session.keys())]
    return redirect(url_for("home"))


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.environ.get("PORT", 5000))
